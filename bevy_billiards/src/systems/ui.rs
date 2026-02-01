//! UI systems for training overlay display
//!
//! Shows generation, fitness, score, and connection status.

use bevy::prelude::*;

use crate::components::{NetworkState, TrainingState};

/// Marker component for the main UI root
#[derive(Component)]
pub struct TrainingUiRoot;

/// Marker for generation text
#[derive(Component)]
pub struct GenerationText;

/// Marker for fitness text
#[derive(Component)]
pub struct FitnessText;

/// Marker for score text
#[derive(Component)]
pub struct ScoreText;

/// Marker for balls remaining text
#[derive(Component)]
pub struct BallsRemainingText;

/// Marker for connection status text
#[derive(Component)]
pub struct ConnectionText;

/// Marker for FPS text
#[derive(Component)]
pub struct FpsText;

/// Colors for the UI
const UI_BG_COLOR: Color = Color::srgba(0.1, 0.1, 0.1, 0.85);
const UI_TEXT_COLOR: Color = Color::srgb(0.9, 0.9, 0.9);
const UI_ACCENT_COLOR: Color = Color::srgb(0.3, 0.8, 0.4);
const UI_WARNING_COLOR: Color = Color::srgb(0.9, 0.6, 0.2);
const UI_ERROR_COLOR: Color = Color::srgb(0.9, 0.3, 0.3);

/// System to spawn the UI
pub fn spawn_ui(mut commands: Commands) {
    // Root container - top-left overlay
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0),
                top: Val::Px(20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(15.0)),
                row_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(UI_BG_COLOR),
            BorderRadius::all(Val::Px(8.0)),
            TrainingUiRoot,
            Name::new("Training UI"),
        ))
        .with_children(|parent| {
            // Title
            parent.spawn((
                Text::new("VIVA NEAT Training"),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(UI_ACCENT_COLOR),
            ));

            // Divider line (using a thin node)
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    margin: UiRect::vertical(Val::Px(5.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.3)),
            ));

            // Generation
            parent.spawn((
                Text::new("Generation: 0"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(UI_TEXT_COLOR),
                GenerationText,
            ));

            // Fitness
            parent.spawn((
                Text::new("Fitness: 0.00"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(UI_TEXT_COLOR),
                FitnessText,
            ));

            // Score
            parent.spawn((
                Text::new("Score: 0"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(UI_TEXT_COLOR),
                ScoreText,
            ));

            // Balls remaining
            parent.spawn((
                Text::new("Balls: 15"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(UI_TEXT_COLOR),
                BallsRemainingText,
            ));

            // Another divider
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    margin: UiRect::vertical(Val::Px(5.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.3)),
            ));

            // Connection status
            parent.spawn((
                Text::new("Disconnected"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(UI_ERROR_COLOR),
                ConnectionText,
            ));

            // FPS counter
            parent.spawn((
                Text::new("FPS: --"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgba(0.7, 0.7, 0.7, 1.0)),
                FpsText,
            ));
        });

    // Help text - bottom right
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(20.0),
                bottom: Val::Px(20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
            BorderRadius::all(Val::Px(5.0)),
            Name::new("Help UI"),
        ))
        .with_children(|parent| {
            let help_lines = [
                "Controls:",
                "Right Mouse - Orbit camera",
                "Middle Mouse - Pan camera",
                "Scroll - Zoom",
                "1-4 - Camera presets",
                "R - Reset balls",
                "Home - Reset camera",
            ];

            for line in help_lines {
                parent.spawn((
                    Text::new(line),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgba(0.8, 0.8, 0.8, 1.0)),
                ));
            }
        });

    info!("UI spawned");
}

/// System to update training stats display
pub fn update_training_ui(
    training_state: Res<TrainingState>,
    mut gen_query: Query<&mut Text, (With<GenerationText>, Without<FitnessText>, Without<ScoreText>, Without<BallsRemainingText>)>,
    mut fitness_query: Query<&mut Text, (With<FitnessText>, Without<GenerationText>, Without<ScoreText>, Without<BallsRemainingText>)>,
    mut score_query: Query<&mut Text, (With<ScoreText>, Without<GenerationText>, Without<FitnessText>, Without<BallsRemainingText>)>,
    mut balls_query: Query<&mut Text, (With<BallsRemainingText>, Without<GenerationText>, Without<FitnessText>, Without<ScoreText>)>,
) {
    if let Ok(mut text) = gen_query.get_single_mut() {
        **text = format!("Generation: {}", training_state.generation);
    }

    if let Ok(mut text) = fitness_query.get_single_mut() {
        **text = format!("Fitness: {:.2}", training_state.fitness);
    }

    if let Ok(mut text) = score_query.get_single_mut() {
        **text = format!("Score: {}", training_state.score);
    }

    if let Ok(mut text) = balls_query.get_single_mut() {
        **text = format!("Balls: {}", training_state.balls_remaining);
    }
}

/// System to update connection status display
pub fn update_connection_ui(
    network_state: Res<NetworkState>,
    mut query: Query<(&mut Text, &mut TextColor), With<ConnectionText>>,
) {
    if let Ok((mut text, mut color)) = query.get_single_mut() {
        if network_state.connected {
            **text = format!("Connected ({})", network_state.messages_received);
            *color = TextColor(UI_ACCENT_COLOR);
        } else if let Some(ref err) = network_state.last_error {
            **text = format!("Error: {}", err.chars().take(30).collect::<String>());
            *color = TextColor(UI_ERROR_COLOR);
        } else {
            **text = "Disconnected".to_string();
            *color = TextColor(UI_WARNING_COLOR);
        }
    }
}

/// Resource for FPS tracking
#[derive(Resource, Default)]
pub struct FpsTracker {
    pub frame_times: Vec<f32>,
    pub last_update: f32,
}

/// System to update FPS display
pub fn update_fps_ui(
    time: Res<Time>,
    mut fps_tracker: ResMut<FpsTracker>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    let dt = time.delta_secs();
    fps_tracker.frame_times.push(dt);

    // Update every 0.5 seconds
    fps_tracker.last_update += dt;
    if fps_tracker.last_update >= 0.5 {
        fps_tracker.last_update = 0.0;

        let avg_dt: f32 = fps_tracker.frame_times.iter().sum::<f32>()
            / fps_tracker.frame_times.len() as f32;
        let fps = 1.0 / avg_dt;

        fps_tracker.frame_times.clear();

        if let Ok(mut text) = query.get_single_mut() {
            **text = format!("FPS: {:.0}", fps);
        }
    }
}

/// Plugin for UI systems
pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpsTracker>()
            .add_systems(Startup, spawn_ui)
            .add_systems(
                Update,
                (update_training_ui, update_connection_ui, update_fps_ui),
            );
    }
}
