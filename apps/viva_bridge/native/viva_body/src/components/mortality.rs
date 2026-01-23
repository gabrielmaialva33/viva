use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce
};
use std::sync::OnceLock;

// THE KEY TO LIFE
// Exists only in RAM. If this process dies, the key is lost forever.
static LIFE_KEY: OnceLock<Aes256Gcm> = OnceLock::new();

pub struct Mortality;

impl Mortality {
    /// Gets the existing Life Key or generates a new one.
    /// In a fresh process, this will generate a new random key.
    fn get_key() -> &'static Aes256Gcm {
        LIFE_KEY.get_or_init(|| {
            eprintln!("[viva_body] GENESIS: Generating volatile AES-256 Life Key. DO NOT DUMP RAM.");
            let key = Aes256Gcm::generate_key(&mut OsRng);
            Aes256Gcm::new(&key)
        })
    }

    /// Encrypts a soul signature (plaintext) using the Life Key.
    /// Returns (ciphertext, nonce).
    pub fn encrypt(plaintext: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
        let key = Self::get_key();
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng); // 96-bits; unique per message

        let ciphertext = key
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("Encryption failed: {}", e))?;

        Ok((ciphertext, nonce.to_vec()))
    }

    /// Decrypts a life token using the Life Key.
    /// If the key has changed (process restart), decryption WILL fail.
    pub fn decrypt(ciphertext: &[u8], nonce: &[u8]) -> Result<Vec<u8>, String> {
        let key = Self::get_key();
        let nonce = Nonce::from_slice(nonce);

        key.decrypt(nonce, ciphertext)
            .map_err(|_| "DECRYPTION_FAILED: The Life Key does not match. The body has perished.".to_string())
    }
}
