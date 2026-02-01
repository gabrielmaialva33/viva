//! WebSocket networking for receiving physics state from VIVA
//!
//! Handles async WebSocket connection and message processing.

use bevy::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::thread;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::components::{BallState, NetworkMessage, NetworkState, PhysicsState, TrainingState};

/// Channel for receiving WebSocket messages in Bevy
#[derive(Resource)]
pub struct NetworkChannel {
    pub receiver: Receiver<NetworkPacket>,
    pub sender: Sender<NetworkCommand>,
}

/// Packets received from the WebSocket server
#[derive(Debug, Clone)]
pub enum NetworkPacket {
    Connected,
    Disconnected,
    Error(String),
    PhysicsUpdate(PhysicsUpdate),
    TrainingUpdate(TrainingUpdate),
}

/// Commands to send to the WebSocket thread
#[derive(Debug, Clone)]
pub enum NetworkCommand {
    Connect(String),
    Disconnect,
    Send(String),
}

/// Physics state update from the server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsUpdate {
    pub timestamp: f64,
    pub balls: Vec<BallData>,
}

/// Individual ball data from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallData {
    pub id: u8,
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub velocity: [f32; 3],
    #[serde(default)]
    pub angular_velocity: [f32; 3],
    #[serde(default = "default_active")]
    pub active: bool,
}

fn default_active() -> bool {
    true
}

/// Training state update from the server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingUpdate {
    pub generation: u32,
    pub fitness: f32,
    pub score: u32,
    pub balls_remaining: u8,
    #[serde(default)]
    pub best_fitness: f32,
    #[serde(default)]
    pub games_played: u32,
}

/// Combined message format from server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "physics")]
    Physics(PhysicsUpdate),
    #[serde(rename = "training")]
    Training(TrainingUpdate),
    #[serde(rename = "reset")]
    Reset,
    #[serde(rename = "pocketed")]
    BallPocketed { ball: u8, pocket: u8 },
}

/// Plugin for WebSocket networking
pub struct NetworkPlugin;

impl Plugin for NetworkPlugin {
    fn build(&self, app: &mut App) {
        // Create channels for communication
        let (packet_tx, packet_rx) = bounded::<NetworkPacket>(100);
        let (command_tx, command_rx) = bounded::<NetworkCommand>(10);

        // Spawn the async networking thread
        let packet_sender = packet_tx.clone();
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            rt.block_on(async move {
                network_loop(packet_sender, command_rx).await;
            });
        });

        app.insert_resource(NetworkChannel {
            receiver: packet_rx,
            sender: command_tx,
        })
        .insert_resource(NetworkState::default())
        .insert_resource(PhysicsState::default())
        .add_event::<NetworkMessage>()
        .add_systems(Startup, auto_connect)
        .add_systems(Update, receive_network_packets);
    }
}

/// Auto-connect to the WebSocket server on startup
fn auto_connect(channel: Res<NetworkChannel>, state: Res<NetworkState>) {
    let _ = channel
        .sender
        .send(NetworkCommand::Connect(state.server_url.clone()));
}

/// Main network loop running in a separate thread
async fn network_loop(packet_tx: Sender<NetworkPacket>, command_rx: Receiver<NetworkCommand>) {
    let mut ws_stream: Option<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    > = None;

    loop {
        // Check for commands
        if let Ok(command) = command_rx.try_recv() {
            match command {
                NetworkCommand::Connect(url) => {
                    info!("Connecting to WebSocket server: {}", url);
                    match connect_async(&url).await {
                        Ok((stream, _)) => {
                            ws_stream = Some(stream);
                            let _ = packet_tx.send(NetworkPacket::Connected);
                            info!("WebSocket connected successfully");
                        }
                        Err(e) => {
                            let _ = packet_tx.send(NetworkPacket::Error(format!(
                                "Connection failed: {}",
                                e
                            )));
                            warn!("WebSocket connection failed: {}", e);
                        }
                    }
                }
                NetworkCommand::Disconnect => {
                    if let Some(mut stream) = ws_stream.take() {
                        let _ = stream.close(None).await;
                    }
                    let _ = packet_tx.send(NetworkPacket::Disconnected);
                }
                NetworkCommand::Send(msg) => {
                    if let Some(ref mut stream) = ws_stream {
                        let _ = stream.send(Message::Text(msg.into())).await;
                    }
                }
            }
        }

        // Read messages from WebSocket
        if let Some(ref mut stream) = ws_stream {
            match tokio::time::timeout(
                tokio::time::Duration::from_millis(10),
                stream.next(),
            )
            .await
            {
                Ok(Some(Ok(msg))) => {
                    if let Message::Text(text) = msg {
                        if let Err(e) = process_message(text.as_str(), &packet_tx) {
                            warn!("Failed to process message: {}", e);
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    let _ = packet_tx.send(NetworkPacket::Error(format!("WebSocket error: {}", e)));
                    ws_stream = None;
                    let _ = packet_tx.send(NetworkPacket::Disconnected);
                }
                Ok(None) => {
                    // Stream ended
                    ws_stream = None;
                    let _ = packet_tx.send(NetworkPacket::Disconnected);
                }
                Err(_) => {
                    // Timeout, no message available
                }
            }
        } else {
            // Not connected, sleep a bit
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
}

/// Process a message from the WebSocket server
fn process_message(text: &str, packet_tx: &Sender<NetworkPacket>) -> Result<(), String> {
    // Try to parse as ServerMessage
    match serde_json::from_str::<ServerMessage>(text) {
        Ok(ServerMessage::Physics(update)) => {
            packet_tx
                .send(NetworkPacket::PhysicsUpdate(update))
                .map_err(|e| e.to_string())?;
        }
        Ok(ServerMessage::Training(update)) => {
            packet_tx
                .send(NetworkPacket::TrainingUpdate(update))
                .map_err(|e| e.to_string())?;
        }
        Ok(ServerMessage::Reset) => {
            // Handle reset if needed
        }
        Ok(ServerMessage::BallPocketed { ball, pocket }) => {
            info!("Ball {} pocketed in pocket {}", ball, pocket);
        }
        Err(_) => {
            // Try parsing as raw physics update for backwards compatibility
            if let Ok(physics) = serde_json::from_str::<PhysicsUpdate>(text) {
                packet_tx
                    .send(NetworkPacket::PhysicsUpdate(physics))
                    .map_err(|e| e.to_string())?;
            } else {
                return Err(format!("Unknown message format: {}", text));
            }
        }
    }
    Ok(())
}

/// System to receive network packets and update resources
fn receive_network_packets(
    channel: Res<NetworkChannel>,
    mut network_state: ResMut<NetworkState>,
    mut physics_state: ResMut<PhysicsState>,
    mut training_state: ResMut<TrainingState>,
) {
    // Process all available packets
    while let Ok(packet) = channel.receiver.try_recv() {
        match packet {
            NetworkPacket::Connected => {
                network_state.connected = true;
                network_state.last_error = None;
                info!("Network connected");
            }
            NetworkPacket::Disconnected => {
                network_state.connected = false;
                info!("Network disconnected");
            }
            NetworkPacket::Error(err) => {
                network_state.last_error = Some(err.clone());
                warn!("Network error: {}", err);
            }
            NetworkPacket::PhysicsUpdate(update) => {
                network_state.messages_received += 1;
                physics_state.timestamp = update.timestamp;
                physics_state.initialized = true;

                // Update ball states
                physics_state.balls.clear();
                for ball_data in update.balls {
                    physics_state.balls.push(BallState {
                        id: ball_data.id,
                        position: Vec3::from_array(ball_data.position),
                        velocity: Vec3::from_array(ball_data.velocity),
                        angular_velocity: Vec3::from_array(ball_data.angular_velocity),
                        active: ball_data.active,
                    });
                }
            }
            NetworkPacket::TrainingUpdate(update) => {
                training_state.generation = update.generation;
                training_state.fitness = update.fitness;
                training_state.score = update.score;
                training_state.balls_remaining = update.balls_remaining;
                training_state.best_fitness = update.best_fitness;
                training_state.games_played = update.games_played;
            }
        }
    }
}

/// Convert physics state to ball positions for rendering
impl PhysicsState {
    pub fn get_ball_position(&self, ball_id: u8) -> Option<Vec3> {
        self.balls
            .iter()
            .find(|b| b.id == ball_id && b.active)
            .map(|b| b.position)
    }

    pub fn is_ball_active(&self, ball_id: u8) -> bool {
        self.balls
            .iter()
            .find(|b| b.id == ball_id)
            .map(|b| b.active)
            .unwrap_or(false)
    }
}
