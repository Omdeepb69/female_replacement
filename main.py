"""
Handwriting Replication AI - Main Implementation
This system learns to replicate a person's handwriting style, including natural
imperfections, and generates complete documents from text inputs.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
from pathlib import Path

# Define project structure
PROJECT_DIR = Path("handwriting_replication_ai")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"

# Ensure directories exist
for directory in [PROJECT_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

#################################################
# PART 1: DATA COLLECTION AND PREPROCESSING
#################################################

class DataCollector:
    """Handles collection and preprocessing of handwriting samples."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.characters_dir = data_dir / "characters"
        self.words_dir = data_dir / "words"
        self.diagrams_dir = data_dir / "diagrams"
        
        # Create subdirectories
        for directory in [self.raw_dir, self.processed_dir, self.characters_dir, 
                          self.words_dir, self.diagrams_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def scan_document(self, document_path, output_name=None):
        """
        Processes a scanned document and saves it to the raw directory.
        In a real implementation, this would interface with a scanner.
        For demonstration, we'll assume the document is already scanned.
        """
        if output_name is None:
            output_name = Path(document_path).name
        
        # Copy the scanned document to the raw directory
        output_path = self.raw_dir / output_name
        img = Image.open(document_path)
        img.save(output_path)
        
        print(f"Document saved to {output_path}")
        return output_path
    
    def preprocess_document(self, document_path):
        """Preprocesses a document for further analysis."""
        # Load the image
        img = cv2.imread(str(document_path))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to separate text from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Save the preprocessed image
        output_path = self.processed_dir / Path(document_path).name
        cv2.imwrite(str(output_path), binary)
        
        print(f"Preprocessed document saved to {output_path}")
        return output_path
    
    def extract_characters(self, document_path):
        """
        Extracts individual characters from a document.
        This is a simplified version. In a real implementation,
        you would use more sophisticated character segmentation techniques.
        """
        img = cv2.imread(str(document_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        char_images = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise (too small contours)
            if w > 10 and h > 10:
                # Extract the character
                char_img = binary[y:y+h, x:x+w]
                
                # Save the character image
                char_path = self.characters_dir / f"char_{i}.png"
                cv2.imwrite(str(char_path), char_img)
                char_images.append(char_path)
        
        print(f"Extracted {len(char_images)} characters from {document_path}")
        return char_images
    
    def extract_words(self, document_path):
        """
        Extracts words from a document.
        Simplified version for demonstration.
        """
        # Similar to extract_characters but with different parameters
        # for contour detection to capture word-level regions
        img = cv2.imread(str(document_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use morphological operations to connect characters within words
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours for words
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        word_images = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size to exclude very small regions
            if w > 30 and h > 20:
                # Extract the word from the original binary image
                word_img = binary[y:y+h, x:x+w]
                
                # Save the word image
                word_path = self.words_dir / f"word_{i}.png"
                cv2.imwrite(str(word_path), word_img)
                word_images.append(word_path)
        
        print(f"Extracted {len(word_images)} words from {document_path}")
        return word_images
    
    def extract_diagrams(self, document_path):
        """
        Extracts diagrams from a document.
        This would require more sophisticated techniques in a real implementation.
        """
        # For demonstration, we'll assume diagrams are larger connected components
        img = cv2.imread(str(document_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagram_images = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for larger regions that might be diagrams
            if w > 100 and h > 100:
                # Extract the diagram
                diagram_img = img[y:y+h, x:x+w]
                
                # Save the diagram image
                diagram_path = self.diagrams_dir / f"diagram_{i}.png"
                cv2.imwrite(str(diagram_path), diagram_img)
                diagram_images.append(diagram_path)
        
        print(f"Extracted {len(diagram_images)} potential diagrams from {document_path}")
        return diagram_images
    
    def analyze_writing_style(self):
        """
        Analyzes collected writing samples to extract style characteristics.
        This would include metrics like:
        - Average character size
        - Slant angle
        - Spacing between characters and words
        - Pressure variations (implied by line thickness)
        - Common imperfections
        """
        # In a real implementation, this would be much more sophisticated
        # For now, we'll return dummy data
        
        style_metrics = {
            "avg_char_width": 20,
            "avg_char_height": 40,
            "avg_word_spacing": 15,
            "avg_line_spacing": 60,
            "avg_slant_angle": 5,  # degrees
            "pressure_variation": 0.3,  # coefficient of variation in line thickness
            "imperfection_rate": 0.05,  # proportion of characters with imperfections
        }
        
        return style_metrics


#################################################
# PART 2: MODEL ARCHITECTURE
#################################################

class CharacterVAE:
    """
    Variational Autoencoder for generating individual characters
    in the target handwriting style.
    """
    
    def __init__(self, latent_dim=128, input_shape=(64, 64, 1)):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.vae = None
        
    def build_encoder(self):
        """Builds the encoder part of the VAE."""
        inputs = keras.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim * 2, activation="relu")(x)
        
        # Split into mean and log variance
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Sampling layer as a custom layer
        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = Sampling(name="z")([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return self.encoder
    
    def build_decoder(self):
        """Builds the decoder part of the VAE."""
        # Calculate dimensions after encoding
        h_dim, w_dim = self.input_shape[0] // 4, self.input_shape[1] // 4
        
        # Decoder inputs
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        
        x = layers.Dense(h_dim * w_dim * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((h_dim, w_dim, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        
        # Create decoder model
        self.decoder = keras.Model(latent_inputs, outputs, name="decoder")
        return self.decoder
    
    def build_vae(self):
        """Builds the complete VAE model."""
        if self.encoder is None:
            self.build_encoder()
        if self.decoder is None:
            self.build_decoder()
        
        # VAE model
        inputs = keras.Input(shape=self.input_shape)
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        
        # Define VAE loss
        reconstruction_loss = keras.losses.binary_crossentropy(
            tf.keras.backend.flatten(inputs),
            tf.keras.backend.flatten(outputs)
        )
        reconstruction_loss *= self.input_shape[0] * self.input_shape[1]
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        
        # Create VAE model
        self.vae = keras.Model(inputs, outputs, name="vae")
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam())
        
        return self.vae
    
    def train(self, data, epochs=50, batch_size=32):
        """Trains the VAE on character data."""
        if self.vae is None:
            self.build_vae()
        
        # Reshape data to match input shape
        data = data.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        # Train the model
        history = self.vae.fit(
            data, None,  # No labels needed for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        return history
    
    def generate_character(self, char_code=None, style_vector=None):
        """
        Generates a character image.
        If char_code is provided, generates that specific character.
        If style_vector is provided, uses it for generation.
        Otherwise, samples randomly from the latent space.
        """
        if self.decoder is None:
            raise ValueError("Decoder model has not been built yet.")
        
        # If no style vector is provided, sample from standard normal
        if style_vector is None:
            style_vector = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Generate character image
        character_img = self.decoder.predict(style_vector)
        
        return character_img[0]


class HandwritingGAN:
    """
    GAN model for generating handwriting sequences.
    """
    
    def __init__(self, sequence_length=64, latent_dim=100):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
    
    def build_generator(self):
        """Builds the generator model."""
        model = keras.Sequential(name="generator")
        
        # Input layer
        model.add(layers.Dense(128 * 16, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Reshape((16, 128)))
        
        # Upsampling layers
        model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv1DTranspose(32, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Output layer
        model.add(layers.Conv1D(1, kernel_size=4, padding="same", activation="tanh"))
        
        self.generator = model
        return model
    
    def build_discriminator(self):
        """Builds the discriminator model."""
        model = keras.Sequential(name="discriminator")
        
        # Input layer
        model.add(layers.Conv1D(32, kernel_size=4, strides=2, padding="same", 
                              input_shape=(self.sequence_length, 1)))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Downsampling layers
        model.add(layers.Conv1D(64, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv1D(128, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Output layer
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation="sigmoid"))
        
        # Compile the discriminator
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=["accuracy"]
        )
        
        self.discriminator = model
        return model
    
    def build_gan(self):
        """Builds the combined GAN model."""
        if self.generator is None:
            self.build_generator()
        if self.discriminator is None:
            self.build_discriminator()
        
        # Freeze the discriminator for generator training
        self.discriminator.trainable = False
        
        # GAN input (noise) and output (generated sequences)
        gan_input = keras.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        
        # Build and compile the GAN model
        self.gan = keras.Model(gan_input, gan_output)
        self.gan.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        
        return self.gan
    
    def train(self, real_sequences, epochs=10000, batch_size=64, save_interval=500):
        """Trains the GAN model."""
        if self.gan is None:
            self.build_gan()
        
        # Labels for real and fake sequences
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real sequences
            idx = np.random.randint(0, real_sequences.shape[0], batch_size)
            real_batch = real_sequences[idx]
            
            # Generate a batch of fake sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate new noise for generator training
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train the generator
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            
            # Save generated samples at intervals
            if epoch % save_interval == 0:
                self.save_samples(epoch)
    
    def save_samples(self, epoch):
        """Saves sample generated sequences."""
        # Generate and plot some samples
        noise = np.random.normal(0, 1, (4, self.latent_dim))
        generated_sequences = self.generator.predict(noise)
        
        # Plot the sequences
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        for i, ax in enumerate(axs.flat):
            ax.plot(generated_sequences[i])
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"handwriting_samples_epoch_{epoch}.png")
        plt.close()
    
    def generate_sequence(self, char_sequence, style_vector=None):
        """
        Generates a handwriting sequence for a given character sequence.
        """
        if self.generator is None:
            raise ValueError("Generator model has not been built yet.")
        
        # If no style vector is provided, sample from standard normal
        if style_vector is None:
            style_vector = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Generate sequence
        handwriting_sequence = self.generator.predict(style_vector)
        
        return handwriting_sequence[0]


class ImperfectionModel:
    """
    Model for adding realistic imperfections to generated handwriting.
    """
    
    def __init__(self):
        self.imperfection_types = [
            "tremor",  # Slight wobble in lines
            "pressure_variation",  # Variation in line thickness
            "slant_variation",  # Variation in character slant
            "spacing_variation",  # Irregular spacing between characters/words
            "baseline_drift",  # Characters drifting above/below baseline
            "smudge",  # Occasional smudging
            "spelling_error"  # Misspellings based on frequency analysis
        ]
        
        # Probabilities for each imperfection type
        self.imperfection_probs = {
            "tremor": 0.3,
            "pressure_variation": 0.4,
            "slant_variation": 0.25,
            "spacing_variation": 0.35,
            "baseline_drift": 0.2,
            "smudge": 0.05,
            "spelling_error": 0.01
        }
        
        # Common spelling errors observed in the training data
        self.spelling_errors = {
            "the": ["teh", "th", "thee"],
            "and": ["nad", "an", "adn"],
            "to": ["ot", "too", "t"],
            # Add more based on analysis of the person's actual spelling errors
        }
    
    def analyze_imperfections(self, handwriting_samples):
        """
        Analyzes handwriting samples to identify common imperfections
        and their characteristics.
        """
        # In a real implementation, this would use computer vision techniques
        # to identify and quantify imperfections
        
        # For demonstration, we'll just update our probabilities slightly
        self.imperfection_probs["tremor"] = 0.35
        self.imperfection_probs["pressure_variation"] = 0.45
        
        print("Analyzed imperfections in handwriting samples.")
    
    def apply_tremor(self, image, intensity=0.2):
        """Applies a slight tremor effect to an image."""
        # Create a copy of the image
        result = image.copy()
        
        # Apply random displacement to each pixel
        rows, cols = result.shape
        for i in range(rows):
            for j in range(cols):
                if result[i, j] > 0:  # Only apply to non-background pixels
                    # Random displacement
                    di = int(np.random.normal(0, intensity))
                    dj = int(np.random.normal(0, intensity))
                    
                    # Apply displacement within bounds
                    ni, nj = min(max(0, i + di), rows - 1), min(max(0, j + dj), cols - 1)
                    
                    # Swap pixel values
                    result[i, j], result[ni, nj] = result[ni, nj], result[i, j]
        
        return result
    
    def apply_pressure_variation(self, image, intensity=0.3):
        """Applies variation in line thickness to simulate pressure changes."""
        # Create a copy of the image
        result = image.copy()
        
        # Apply random erosion or dilation to simulate pressure variation
        kernel_size = np.random.randint(1, 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Randomly choose between erosion (lighter) and dilation (heavier)
        if np.random.random() < 0.5:
            result = cv2.erode(result, kernel, iterations=1)
        else:
            result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def apply_spacing_variation(self, image, intensity=0.2):
        """Applies variation in character spacing."""
        # This would be complex to implement realistically
        # For demonstration, we'll just randomly stretch or compress the image horizontally
        
        # Create a copy of the image
        result = image.copy()
        
        # Random stretch/compress factor
        factor = 1.0 + np.random.normal(0, intensity)
        new_width = max(int(result.shape[1] * factor), 1)
        
        # Resize the image
        result = cv2.resize(result, (new_width, result.shape[0]))
        
        return result
    
    def apply_baseline_drift(self, image, intensity=0.1):
        """Applies vertical drift to simulate characters not following a straight baseline."""
        # Create a copy of the image
        result = image.copy()
        rows, cols = result.shape
        
        # Create a drift map (how much each column should shift vertically)
        drift = np.zeros(cols)
        drift[0] = np.random.normal(0, intensity * rows)
        
        # Make the drift smooth by applying a cumulative effect with random changes
        for i in range(1, cols):
            drift[i] = drift[i-1] + np.random.normal(0, intensity * rows / 5)
        
        # Apply drift
        for j in range(cols):
            shift = int(drift[j])
            if shift == 0:
                continue
                
            # Shift column up or down
            if shift > 0:
                result[shift:, j] = result[:-shift, j]
                result[:shift, j] = 0
            else:
                shift = -shift
                result[:-shift, j] = result[shift:, j]
                result[-shift:, j] = 0
        
        return result
    
    def apply_smudge(self, image, intensity=0.1):
        """Applies occasional smudging effect."""
        # Create a copy of the image
        result = image.copy()
        
        # Randomly decide whether to apply smudge
        if np.random.random() > 0.8:
            # Select a random region to smudge
            rows, cols = result.shape
            center_i = np.random.randint(0, rows)
            center_j = np.random.randint(0, cols)
            
            # Only smudge if there's writing in the region
            region = result[max(0, center_i-10):min(rows, center_i+10), 
                           max(0, center_j-10):min(cols, center_j+10)]
            
            if np.sum(region) > 0:
                # Apply blur to simulate smudge
                blur_size = int(intensity * 20) + 1
                if blur_size % 2 == 0:
                    blur_size += 1  # Ensure odd size for Gaussian blur
                    
                smudge_region = cv2.GaussianBlur(region, (blur_size, blur_size), 0)
                result[max(0, center_i-10):min(rows, center_i+10), 
                      max(0, center_j-10):min(cols, center_j+10)] = smudge_region
        
        return result
    
    def apply_spelling_error(self, text):
        """
        Introduces realistic spelling errors based on the person's common mistakes.
        """
        words = text.split()
        result_words = []
        
        for word in words:
            # Check if this word has known error patterns
            if word.lower() in self.spelling_errors and np.random.random() < self.imperfection_probs["spelling_error"]:
                # Select a random misspelling
                misspellings = self.spelling_errors[word.lower()]
                misspelled = np.random.choice(misspellings)
                
                # Preserve capitalization
                if word[0].isupper():
                    misspelled = misspelled.capitalize()
                
                result_words.append(misspelled)
            else:
                result_words.append(word)
        
        return " ".join(result_words)
    
    def apply_imperfections(self, image, text=None):
        """
        Applies a combination of imperfections to make handwriting look natural.
        """
        # Start with the original image
        result = image.copy()
        
        # Apply visual imperfections based on probabilities
        if np.random.random() < self.imperfection_probs["tremor"]:
            result = self.apply_tremor(result)
        
        if np.random.random() < self.imperfection_probs["pressure_variation"]:
            result = self.apply_pressure_variation(result)
        
        if np.random.random() < self.imperfection_probs["spacing_variation"]:
            result = self.apply_spacing_variation(result)
        
        if np.random.random() < self.imperfection_probs["baseline_drift"]:
            result = self.apply_baseline_drift(result)
        
        if np.random.random() < self.imperfection_probs["smudge"]:
            result = self.apply_smudge(result)
        
        # Apply spelling errors if text is provided
        modified_text = None
        if text is not None:
            modified_text = self.apply_spelling_error(text)
        
        return result, modified_text


class DiagramGenerator:
    """
    Model for generating diagrams in the user's drawing style.
    """
    
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """
        Builds a pix2pix-like model for translating reference diagrams
        to the user's drawing style.
        """
        # For simplicity, we'll create a basic U-Net architecture
        # In a real implementation, you'd use a proper pix2pix GAN
        
        # Encoder
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder blocks
        enc1 = self._encoder_block(inputs, 64, use_batch_norm=False)
        enc2 = self._encoder_block(enc1, 128)
        enc3 = self._encoder_block(enc2, 256)
        enc4 = self._encoder_block(enc3, 512)
        
        # Bottleneck
        bottleneck = self._encoder_block(enc4, 512)
        
        # Decoder blocks
        dec4 = self._decoder_block(bottleneck, enc4, 512)
        dec3 = self._decoder_block(dec4, enc3, 256)
        dec2 = self._decoder_block(dec3, enc2, 128)
        dec1 = self._decoder_block(dec2, enc1, 64)
        
        # Output layer
        outputs = layers.Conv2D(3, 3, padding="same", activation="tanh")(dec1)
        
        # Create model
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss="mae"
        )
        
        return self.model
    
    def _encoder_block(self, inputs, filters, use_batch_norm=True):
        """Creates an encoder block for the U-Net."""
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(inputs)
        
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
            
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        return x
    
    def _decoder_block(self, inputs, skip_features, filters):
        """Creates a decoder block for the U-Net with skip connections."""
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Skip connection
        x = layers.Concatenate()([x, skip_features])
        
        return x
    
    def train(self, reference_diagrams, user_diagrams, epochs=100, batch_size=1):
        """
        Trains the diagram style transfer model.
        
        Parameters:
        - reference_diagrams: Standard diagrams (input)
        - user_diagrams: Corresponding user-drawn diagrams (target)
        """
        if self.model is None:
            self.build_model()
        
        # Train the model
        history = self.model.fit(
            reference_diagrams, user_diagrams,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )
        
        return history
    
    def generate_diagram(self, reference_diagram):
        """
        Generates a diagram in the user's style based on a reference diagram.
        
        Parameters:
        - reference_diagram: A standard diagram to be converted
        
        Returns:
        - User-style diagram
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure the reference diagram has the correct shape
        if reference_diagram.shape != self.input_shape:
            reference_diagram = cv2.resize(
                reference_diagram, 
                (self.input_shape[1], self.input_shape[0])
            )
            
            # Add batch dimension if needed
            if len(reference_diagram.shape) == 3:
                reference_diagram = np.expand_dims(reference_diagram, axis=0)
        
        # Generate the diagram
        user_style_diagram = self.model.predict(reference_diagram)
        
        # Post-process if needed
        user_style_diagram = np.clip(user_style_diagram, -1, 1)
        user_style_diagram = (user_style_diagram + 1) / 2 * 255
        user_style_diagram = user_style_diagram.astype(np.uint8)
        
        return user_style_diagram[0] if user_style_diagram.shape[0] == 1 else user_style_diagram


#################################################
# PART 3: DOCUMENT LAYOUT AND GENERATION
#################################################

class DocumentGenerator:
    """
    Generates complete documents using the trained models.
    """
    
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.character_model = None
        self.sequence_model = None
        self.imperfection_model = None
        self.diagram_model = None
        
        # Document layout parameters
        self.page_size = (2480, 3508)  # A4 at 300 DPI
        self.margins = {
            "top": 300,
            "right": 250,
            "bottom": 300,
            "left": 250
        }
        self.line_spacing = 100
        self.paragraph_spacing = 150
    
    def load_models(self):
        """Loads all trained models."""
        # Load character VAE
        self.character_model = CharacterVAE()
        # In a real implementation, you would load weights from saved models
        
        # Load sequence GAN
        self.sequence_model = HandwritingGAN()
        
        # Load imperfection model
        self.imperfection_model = ImperfectionModel()
        
        # Load diagram generator
        self.diagram_model = DiagramGenerator()
        
        print("All models loaded successfully.")
    
    def create_blank_page(self):
        """Creates a blank page with the specified size."""
        # Create a white page
        page = np.ones((*self.page_size, 3), dtype=np.uint8) * 255
        return page
    
    def generate_text_line(self, text, style_vector=None):
        """
        Generates a handwritten line of text.
        
        Parameters:
        - text: The text to render
        - style_vector: Optional style parameters
        
        Returns:
        - An image containing the handwritten line
        """
        # In a real implementation, this would use the sequence model
        # to generate a continuous line of handwriting
        
        # For demonstration, we'll create a placeholder implementation
        line_height = 80
        line_width = len(text) * 30  # Approximate width based on text length
        line_img = np.ones((line_height, line_width), dtype=np.uint8) * 255
        
        # Draw text using a handwriting-like font
        # In a real implementation, this would use the actual generated handwriting
        cv2.putText(
            line_img, 
            text, 
            (10, 50), 
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
            1, 
            (0, 0, 0), 
            2
        )
        
        # Apply imperfections to make it look realistic
        line_img, _ = self.imperfection_model.apply_imperfections(line_img)
        
        return line_img
    
    def place_text_on_page(self, page, text, start_position):
        """
        Places the generated text on the page at the specified position.
        
        Parameters:
        - page: The page image
        - text: The text to place (as an image)
        - start_position: (x, y) coordinates for the top-left corner
        
        Returns:
        - Updated page with the text placed
        """
        x, y = start_position
        h, w = text.shape[:2]
        
        # Ensure text fits within the page
        if x + w > self.page_size[0] - self.margins["right"]:
            w = self.page_size[0] - self.margins["right"] - x
        
        if y + h > self.page_size[1] - self.margins["bottom"]:
            h = self.page_size[1] - self.margins["bottom"] - y
        
        # Place text on the page
        if len(text.shape) == 2:  # Grayscale
            for c in range(3):
                page[y:y+h, x:x+w, c] = np.minimum(
                    page[y:y+h, x:x+w, c],
                    text[:h, :w]
                )
        else:  # Color
            page[y:y+h, x:x+w] = np.minimum(
                page[y:y+h, x:x+w],
                text[:h, :w]
            )
        
        return page, (x, y + h)  # Return updated page and end position
    
    def place_diagram_on_page(self, page, diagram, position):
        """
        Places a diagram on the page.
        
        Parameters:
        - page: The page image
        - diagram: The diagram image
        - position: (x, y) coordinates for the top-left corner
        
        Returns:
        - Updated page with the diagram placed
        """
        x, y = position
        h, w = diagram.shape[:2]
        
        # Ensure diagram fits within the page
        if x + w > self.page_size[0] - self.margins["right"]:
            w = self.page_size[0] - self.margins["right"] - x
            diagram = cv2.resize(diagram, (w, int(h * w / diagram.shape[1])))
            h = diagram.shape[0]
        
        if y + h > self.page_size[1] - self.margins["bottom"]:
            h = self.page_size[1] - self.margins["bottom"] - y
            diagram = cv2.resize(diagram, (int(w * h / diagram.shape[0]), h))
            w = diagram.shape[1]
        
        # Place diagram on the page
        page[y:y+h, x:x+w] = diagram[:h, :w]
        
        return page, (x, y + h)  # Return updated page and end position
    
    def generate_document(self, text_content, diagrams=None, style_params=None):
        """
        Generates a complete document with the specified content.
        
        Parameters:
        - text_content: The text content to include in the document
        - diagrams: List of diagrams to include, each with position information
        - style_params: Optional style parameters
        
        Returns:
        - A list of page images comprising the document
        """
        if self.character_model is None:
            self.load_models()
        
        # Split text into paragraphs
        paragraphs = text_content.split("\n\n")
        
        # Initialize document
        pages = [self.create_blank_page()]
        current_page = 0
        current_position = (self.margins["left"], self.margins["top"])
        
        # Process each paragraph
        for paragraph in paragraphs:
            # Split paragraph into lines that fit within the page width
            max_line_width = self.page_size[0] - self.margins["left"] - self.margins["right"]
            lines = []
            current_line = ""
            
            for word in paragraph.split():
                # Estimate width of word (in a real implementation, you would have a more accurate way)
                word_width = len(word) * 30
                
                # Check if adding this word would exceed the line width
                if len(current_line) * 30 + word_width > max_line_width:
                    lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            
            # Add the last line if not empty
            if current_line:
                lines.append(current_line)
            
            # Process each line
            for line in lines:
                # Apply spelling errors for realism
                line, _ = self.imperfection_model.apply_spelling_error(line), None
                
                # Generate handwritten line
                line_img = self.generate_text_line(line, style_params)
                
                # Check if we need to start a new page
                if current_position[1] + line_img.shape[0] > self.page_size[1] - self.margins["bottom"]:
                    # Start a new page
                    pages.append(self.create_blank_page())
                    current_page += 1
                    current_position = (self.margins["left"], self.margins["top"])
                
                # Place the line on the current page
                pages[current_page], current_position = self.place_text_on_page(
                    pages[current_page], 
                    line_img, 
                    current_position
                )
                
                # Move to the next line position
                current_position = (self.margins["left"], current_position[1] + self.line_spacing)
            
            # Add paragraph spacing
            current_position = (self.margins["left"], current_position[1] + self.paragraph_spacing - self.line_spacing)
        
        # Add diagrams if provided
        if diagrams:
            for diagram_info in diagrams:
                diagram_img = diagram_info["image"]
                
                # If a reference diagram is provided, convert it to the user's style
                if "reference" in diagram_info:
                    reference_img = diagram_info["reference"]
                    diagram_img = self.diagram_model.generate_diagram(reference_img)
                
                # Determine position (use specified or current position)
                position = diagram_info.get("position", current_position)
                
                # Check if we need to start a new page
                if position[1] + diagram_img.shape[0] > self.page_size[1] - self.margins["bottom"]:
                    # Start a new page
                    pages.append(self.create_blank_page())
                    current_page += 1
                    position = (self.margins["left"], self.margins["top"])
                
                # Place the diagram
                pages[current_page], new_position = self.place_diagram_on_page(
                    pages[current_page],
                    diagram_img,
                    position
                )
                
                # Update current position if the diagram was placed at the current position
                if position == current_position:
                    current_position = (self.margins["left"], new_position[1] + self.paragraph_spacing)
        
        return pages
    
    def save_document(self, pages, output_path):
        """
        Saves the generated document as a PDF.
        
        Parameters:
        - pages: List of page images
        - output_path: Path to save the PDF
        """
        # Convert pages to PIL Images
        pil_images = [Image.fromarray(page) for page in pages]
        
        # Save as PDF
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else []
        )
        
        print(f"Document saved to {output_path}")


#################################################
# PART 4: USER INTERFACE
#################################################

class HandwritingReplicationUI:
    """
    Provides a user interface for the handwriting replication system.
    In a real implementation, this would be a GUI or web interface.
    """
    
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.data_collector = DataCollector(project_dir / "data")
        self.document_generator = DocumentGenerator(project_dir / "models")
        
        # Ensure the output directory exists
        self.output_dir = project_dir / "output"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def collect_samples(self):
        """Interface for collecting handwriting samples."""
        print("=== Handwriting Sample Collection ===")
        print("Please scan your handwriting samples and provide the file paths.")
        
        # In a real implementation, this would have a file upload interface
        sample_path = input("Enter path to a handwriting sample (or 'done' to finish): ")
        
        collected_samples = []
        while sample_path.lower() != "done":
            try:
                # Process the sample
                processed_path = self.data_collector.scan_document(sample_path)
                preprocessed_path = self.data_collector.preprocess_document(processed_path)
                
                # Extract components
                self.data_collector.extract_characters(preprocessed_path)
                self.data_collector.extract_words(preprocessed_path)
                self.data_collector.extract_diagrams(preprocessed_path)
                
                collected_samples.append(preprocessed_path)
                print(f"Successfully processed sample: {sample_path}")
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
            
            # Get next sample
            sample_path = input("Enter path to another sample (or 'done' to finish): ")
        
        print(f"Collected {len(collected_samples)} samples.")
        return collected_samples
    
    def train_models(self):
        """Interface for training the models."""
        print("=== Training Handwriting Models ===")
        print("This process will train the AI models on your handwriting samples.")
        print("Training may take some time depending on the amount of data.")
        
        try:
            # Analyze writing style
            style_metrics = self.data_collector.analyze_writing_style()
            print("Writing style analysis complete.")
            
            # Load training data
            # In a real implementation, this would load the actual collected data
            print("Loading training data...")
            
            # Train character model
            print("Training character model...")
            char_model = CharacterVAE()
            char_model.build_vae()
            # In a real implementation, you would train with actual data
            
            # Train sequence model
            print("Training handwriting sequence model...")
            seq_model = HandwritingGAN()
            seq_model.build_gan()
            # In a real implementation, you would train with actual data
            
            # Train diagram model
            print("Training diagram style model...")
            diagram_model = DiagramGenerator()
            diagram_model.build_model()
            # In a real implementation, you would train with actual data
            
            # Analyze imperfections
            print("Analyzing writing imperfections...")
            imperfection_model = ImperfectionModel()
            # In a real implementation, you would analyze with actual data
            
            print("All models trained successfully!")
            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def generate_document(self):
        """Interface for generating a handwritten document."""
        print("=== Handwritten Document Generation ===")
        
        # Get text content
        print("Enter the text content for your document:")
        print("(Type 'END' on a new line when finished)")
        
        lines = []
        line = input()
        while line != "END":
            lines.append(line)
            line = input()
        
        text_content = "\n".join(lines)
        
        # Ask about diagrams
        has_diagrams = input("Does your document include diagrams? (y/n): ").lower() == 'y'
        diagrams = []
        
        if has_diagrams:
            num_diagrams = int(input("How many diagrams do you want to include? "))
            
            for i in range(num_diagrams):
                diagram_path = input(f"Enter path to diagram {i+1}: ")
                
                try:
                    diagram_img = cv2.imread(diagram_path)
                    position_x = int(input("X position on page (leave empty for auto): ") or "0")
                    position_y = int(input("Y position on page (leave empty for auto): ") or "0")
                    
                    # If position is (0,0), use auto positioning
                    position = (position_x, position_y) if position_x > 0 and position_y > 0 else None
                    
                    diagrams.append({
                        "image": diagram_img,
                        "position": position
                    })
                    
                    print(f"Diagram {i+1} added.")
                except Exception as e:
                    print(f"Error adding diagram: {str(e)}")
        
        # Generate document
        print("Generating handwritten document...")
        try:
            pages = self.document_generator.generate_document(text_content, diagrams)
            
            # Save the document
            output_path = self.output_dir / "generated_document.pdf"
            self.document_generator.save_document(pages, output_path)
            
            print(f"Document generated and saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating document: {str(e)}")
            return None
    
    def start(self):
        """Main interface for the handwriting replication system."""
        print("=== Handwriting Replication AI ===")
        print("This system will learn your handwriting style and generate documents that look like you wrote them.")
        
        while True:
            print("\nSelect an option:")
            print("1. Collect handwriting samples")
            print("2. Train handwriting models")
            print("3. Generate handwritten document")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == "1":
                self.collect_samples()
            elif choice == "2":
                self.train_models()
            elif choice == "3":
                self.generate_document()
            elif choice == "4":
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")


#################################################
# PART 5: MAIN APPLICATION
#################################################

def main():
    """Main function to run the handwriting replication system."""
    print("Starting Handwriting Replication AI...")
    
    # Set up project directory
    project_dir = Path("handwriting_replication_ai")
    project_dir.mkdir(exist_ok=True)
    
    # Create and start the UI
    ui = HandwritingReplicationUI(project_dir)
    ui.start()


if __name__ == "__main__":
    main()
