import nibabel as nib
from PIL import Image
import random
from tensorflow.keras import layers, models
import numpy as np
import cv2
import tarfile
import os
import matplotlib.pyplot as plt
import pydicom
import csv

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.losses import binary_crossentropy

import math
import os
from scipy.ndimage import rotate,zoom
import copy
from scipy.ndimage import distance_transform_edt as distance
from keras import backend as K
from model_architectures import Architectures

class TrainModel():
    def __init__(self):
        self.architectures = Architectures()
        self.configure_gpu()
        def dice_loss( y_true, y_pred):
          smooth = 1e-6  
          y_true_f = tf.reshape(y_true, [-1])
          y_pred_f = tf.reshape(y_pred, [-1])
          intersection = tf.reduce_sum(y_true_f * y_pred_f)
          return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f)\
          + tf.reduce_sum(y_pred_f) + smooth)

        def combined_dice_bce_loss( y_true, y_pred, alpha=0.5):
            dice_loss = dice_loss(y_true, y_pred)
            bce_loss = binary_crossentropy(y_true, y_pred)
            combined_loss = (alpha * dice_loss) + ((1 - alpha) * bce_loss)
            return combined_loss

        def apply_threshold(preds, threshold=0.5):
            return tf.cast(tf.greater(preds, threshold), tf.float64)

        def boundary_loss(self,y_true, y_pred):
          sobel_filter = tf.image.sobel_edges
          y_true_edges = sobel_filter(y_true)
          y_pred_edges = sobel_filter(y_pred)
          loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

          return loss

        
        def loss(y_true, y_pred):
            pass

        def weighted_boundary_loss(y_true, y_pred):
            pass


        custom_objects = {
            'combined_dice_bce_loss':combined_dice_bce_loss,
            'dice_loss':dice_loss,
            'boundary_loss':boundary_loss,
            'combined_loss':self.combined_loss,
            'weighted_boundary_loss':weighted_boundary_loss,
            "loss":loss
        }

        self.mask_model = load_model(\
            '/data/pnlx/projects/mysell_masking_cnn/final_results/models/attention_unet_stage_one_final_refined_dice_best_only.h5',\
            custom_objects=custom_objects)
        
        #self.mask_model = self.architectures.deep_atrous_attn_unet()

        self.mask_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
            )
        
        def generator_loss(mask_loss, orig_input, generated_input, mask, disc_loss):
            #normalized_mse_loss = self.percentage_mae_loss(orig_input, generated_input,mask) / .167

            #new_loss = self.grey_white_matter_loss(orig_input, generated_input,mask)
            #print(new_loss)

            #print(f'mse {normalized_mse_loss}')

            capped_mask_loss = tf.minimum(mask_loss, 1.0)
            #capped_disc_loss = tf.minimum(disc_loss, 1.0)
            ssim_loss = self.calculate_ssim_loss(orig_input, generated_input)

            capped_mask_loss = tf.cast(capped_mask_loss, tf.float32)
            #new_loss = tf.cast(new_loss, tf.float32)

            #loss = (2 * (1-capped_mask_loss) ) + ((1-capped_disc_loss) *.32) + (normalized_mse_loss * 1.1)
            loss = ((1-capped_mask_loss) ) + (disc_loss * 2) 

            return loss

        def masker_loss(real_mask, generated_mask):
            total_loss = self.combined_loss(real_mask,generated_mask)
            return total_loss
        
        def discriminator_loss(y_true, y_pred, mask):
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) 
            mask_sum = tf.reduce_sum(mask)

            loss_condition = tf.cond(mask_sum > 0,
                            lambda: bce_loss(y_true, y_pred),
                            lambda: bce_loss(y_true, y_pred)/10)

            return loss_condition


        self.generator_loss_fn = generator_loss
        self.masker_loss_fn = masker_loss

        self.discrimnator_loss_fn = discriminator_loss
    
    def configure_gpu(self, gpu_to_use : int = 0) -> None:
        """
        Sets up GPU that tensorflow 
        will use to train model.

        Parameters
        -------------
        gpu_to_use: int
            index of GPU
        """
        gpus = tf.config.experimental. \
        list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40960)]
                    )
                tf.config.experimental.set_visible_devices( \
                gpus[gpu_to_use], 'GPU')
            except RuntimeError as e:
                print(e)
    def calculate_ssim_loss(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        
        # Compute SSIM
        ssim_values = tf.image.ssim(y_true, y_pred, max_val=1.0)
        
        # Calculate the SSIM loss
        ssim_loss = 1 - tf.reduce_mean(ssim_values)
        
        return ssim_loss
    def run_script(self, load_old_model : bool = False) -> None:
        """
        Defines architecture and 
        calls training functions.

        Parameters
        -------------
        load_old_model: bool
            If set to True,
            an existing model will be loaded
        """

        if load_old_model:
            self.generator = self.load_trained_model(\
            '/data/pnlx/projects/mysell_masking_cnn/final_results/models/attention_unet_stage_one_best_only.h5')
            self.generator.compile(
                loss=self.combined_loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5),
                metrics=[self.dice_loss, self.weighted_boundary_loss]
            )
        else:
            self.generator = self.architectures.atrous_attn_unet()
            #self.generator = self.load_trained_model(\
            #'/data/pnlx/projects/mysell_masking_cnn/final_project/GAN_aug_test/checkpoints/generator_full_aug_epoch_1_step_70400.h5')
            self.generator.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
            )

            self.discriminator = self.architectures.atrous_attn_unet_discriminator()

            self.discriminator.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
            )


            self.masker =  self.mask_model

            print(self.generator.summary())

        self.train()

    def load_trained_model(self,path_to_model : str) -> tf.keras.models.Model:
        """
        Loads a previously trained model
        in order to continue training.

        Parameters 
        -------------
        path_to_model: str
            path to the saved model

        Returns
        ----------
        model : tf.keras.models.Model 
            loaded model
        """
        def dice_loss( y_true, y_pred):
          smooth = 1e-6  
          y_true_f = tf.reshape(y_true, [-1])
          y_pred_f = tf.reshape(y_pred, [-1])
          intersection = tf.reduce_sum(y_true_f * y_pred_f)
          return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f)\
          + tf.reduce_sum(y_pred_f) + smooth)

        def combined_dice_bce_loss( y_true, y_pred, alpha=0.5):
            dice_loss = dice_loss(y_true, y_pred)
            bce_loss = binary_crossentropy(y_true, y_pred)
            combined_loss = (alpha * dice_loss) + ((1 - alpha) * bce_loss)
            return combined_loss

        def apply_threshold(preds, threshold=0.5):
            return tf.cast(tf.greater(preds, threshold), tf.float64)

        def boundary_loss(self,y_true, y_pred):
          sobel_filter = tf.image.sobel_edges
          y_true_edges = sobel_filter(y_true)
          y_pred_edges = sobel_filter(y_pred)
          loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

          return loss

        def combined_loss(y_true, y_pred, alpha=0.3, beta=0.3, gamma=1.5):
          bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
          d_loss = self.dice_loss(y_true, y_pred)
          boundary = self.boundary_loss(y_true, y_pred)

          total_loss = (alpha * bce) + (beta * d_loss) + (gamma * boundary)
          
          return total_loss
        
        def loss(y_true, y_pred):
            pass

        def weighted_boundary_loss(y_true, y_pred):
            pass


        custom_objects = {
            'combined_dice_bce_loss':combined_dice_bce_loss,
            'dice_loss':dice_loss,
            'boundary_loss':boundary_loss,
            'combined_loss':combined_loss,
            'weighted_boundary_loss':weighted_boundary_loss,
            "loss":loss
        }

        
        def loss(y_true,y_pred):
            pass
       
        model = load_model(\
        path_to_model,\
        custom_objects=custom_objects)

        return model

    def percentage_mae_loss(self,y_true, y_pred, mask):
        epsilon = 1e-10

        to_white = tf.square(tf.abs(y_pred - y_true) / tf.abs(1 - y_true + epsilon))
        to_black = tf.square(tf.abs(y_pred - y_true) / tf.abs(0 - y_true + epsilon))
        fin_res = tf.where(y_pred < y_true, to_black, to_white)

        fin_res = tf.where(mask > 0, fin_res * 3, fin_res)

        masked_y_true = tf.boolean_mask(y_true, mask > 0)
        masked_y_pred = tf.boolean_mask(y_pred, mask > 0)
        valid_mask = tf.reduce_sum(mask) > 0  
        std_y_true = tf.cond(
            valid_mask, 
            lambda: tf.math.reduce_std(masked_y_true), 
            lambda: tf.constant(0.0)
        )

        std_y_pred = tf.cond(
            valid_mask, 
            lambda: tf.math.reduce_std(masked_y_pred), 
            lambda: tf.constant(0.0)
        )
        std_diff = tf.abs(std_y_true - std_y_pred)

        loss = fin_res + (5 * std_diff)
        
        return tf.reduce_mean(tf.abs(loss))

    
    def mse_loss(self, y_true, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss
        between the prediction and target tensors.

        Args:
        pred (tf.Tensor): Predicted tensor.
        target (tf.Tensor): Ground truth tensor.

        Returns:
        tf.Tensor: MSE loss.
        """
        
        mse_loss_value = tf.reduce_mean(tf.square(y_pred - y_true))
        return mse_loss_value


    def training_generator(
        self, data_dir : str, batch_size : int,
        first_sub : int, last_sub : int
    ) -> tuple:
        """
        Function to read data from each subject folder
        to be fed into the neural network as a training
        set or validation set. 

        Parameters
        -----------------
        data_dir : str
            directory to subject folders 
        batch_size : int
            batch size being used for training
        first_sub : int
            minimum subject in range of subjects being used 
        last_sub : int 
            maximum subject in range of subjects being used 

        Yields
        ------------------
        combined_batch : tuple
            Tuple containing a batch of MRI Data
        """

        subjects = [os.path.join(data_dir, 'subject_' + str(i)) for i in range(first_sub, last_sub)]
        while True:
            np.random.shuffle(subjects)
            for subject in subjects:
                mri_path = os.path.join(subject, 'mri_array.npy')
                mask_path = os.path.join(subject, 'mask_array.npy')
                mri_data = np.load(mri_path)
                mask_data = np.load(mask_path)
                permutation = np.random.permutation(len(mri_data))
                mri_data = mri_data[permutation]
                mask_data = mask_data[permutation]
                for i in range(0, len(mri_data), batch_size):
                    mri_batch = mri_data[i:i + batch_size].astype(np.float32)
                    mask_batch = mask_data[i:i + batch_size].astype(np.float32)
                    combined_batch = (mri_batch, mask_batch)
                    yield combined_batch

    def dice_loss(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        """
        Loss function that calculates
        overlap between ground truth 
        masks and predicted masks.

        Parameters
        ----------------
        y_true : tf.Tensor
            ground truth masks
        y_pred : tf.Tensor
            predicted mask

        Returns
        ------------
        dice_score : tf.Tensor
            Dice coefficient of
            predicted and ground
            truth masks
        """
    
        smooth = 1e-6
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (\
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_score = 1 - tf.clip_by_value(dice_score, 0, 1) 
        dice_score = dice_score 

        return dice_score
    

    def combined_loss(self, y_true, y_pred, bce_weight : float = 0.25,
        dice_weight : float = 0.5
    ) -> tf.Tensor:
        """
        Loss function that combined binary cross-entropy,
        dice coefficient, and boundary loss.

        Parameters
        ------------------
        y_true : tf.Tensor
            ground truth mask
        y_pred : tf.Tensor
            predicted mask
        bce_weight : float
            weight added to binary cross-entropy
        dice_weight : float
            weight added to dice coefficient 
        bound_weight : float
            weight added to boundary loss 

        Returns
        ----------------
        total_loss : tf.Tensor
            calculated combined loss between
            predicted mask and ground truth 
        """

        bce = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25,gamma=2.0,
        from_logits=False)(y_true, y_pred)
        d_loss = self.weighted_dice_loss(y_true, y_pred)

        bce = tf.reduce_mean(bce)
        d_loss = tf.reduce_mean(d_loss)
        #boundary_loss =tf.reduce_mean(boundary_loss)
        
        total_loss = (bce_weight * bce) + (dice_weight * d_loss)

        return total_loss
    

    def grey_white_matter_loss(self, y_true, y_pred, mask):
        if tf.reduce_sum(mask) == 0:  # No valid mask positions
            return tf.constant(0.0)
        

        def replace_with_order_tf_masked(tensor, mask):
            # Flatten the tensor and mask to 1D
            flat_tensor = tf.reshape(tensor, [-1])
            flat_mask = tf.reshape(mask, [-1])

            masked_values = tf.boolean_mask(flat_tensor, flat_mask > 0)

            # Get the order/rank of the selected elements
            order = tf.argsort(masked_values)
            
            # Generate the ranks based on the order
            ranks = tf.argsort(order)
            
            # Create a rank tensor of the same shape as the original tensor, initialized with original values
            rank_tensor = tf.identity(flat_tensor)
            
            # Update only the positions where the mask is greater than 0 with the ranked values
            indices = tf.where(flat_mask > 0)
            rank_tensor = tf.tensor_scatter_nd_update(rank_tensor, indices, tf.cast(ranks, flat_tensor.dtype))
            
            # Reshape the rank tensor back to the original shape of the input tensor
            rank_tensor = tf.reshape(rank_tensor, tf.shape(tensor))
            
            return rank_tensor
        
        def normalize_tensor(tensor):
            # Find the maximum value of the tensor
            max_value = tf.reduce_max(tensor)
            
            # Avoid division by zero by checking if max_value is not zero
            normalized_tensor = tf.cond(max_value > 0, lambda: tensor / max_value, lambda: tensor)
            
            return normalized_tensor

        def apply_mask(tensor, mask):
            # Ensure that the mask is a binary mask (either 0 or 1)
            binary_mask = tf.cast(mask > 0, tensor.dtype)
            
            # Multiply the tensor with the binary mask element-wise
            masked_tensor = tensor * binary_mask
            
            return masked_tensor
        def replace_with_order_tf(tensor):
            # Flatten the tensor to a 1D array
            flat_tensor = tf.reshape(tensor, [-1])
            
            # Get the order/rank of each element (tf.argsort for sorting indices)
            order = tf.argsort(flat_tensor)
            
            # Generate the ranks based on the order
            ranks = tf.argsort(order)
            
            # Reshape the rank tensor back to the original shape
            rank_tensor = tf.reshape(ranks, tf.shape(tensor))
            
            return rank_tensor

        """y_true = replace_with_order_tf_masked(y_true, mask)
        y_pred = replace_with_order_tf_masked(y_pred,mask)
        y_pred = normalize_tensor(y_pred)
        y_true = normalize_tensor(y_true)"""
        y_true = apply_mask(y_true,mask)
        y_pred = apply_mask(y_pred,mask)
        return tf.reduce_mean(tf.square(y_pred - y_true))
        y_true = replace_with_order_tf(y_true)
        y_pred = replace_with_order_tf(y_pred)
        y_pred = normalize_tensor(y_pred)
        y_true = normalize_tensor(y_true)
        mse_loss_value = tf.reduce_mean(tf.square(y_pred - y_true))


        return mse_loss_value



    def weighted_dice_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, weight_fn: float = 3.0, weight_fp: float = 1.0) -> tf.Tensor:
        """
        Weighted Dice loss function that penalizes false negatives more than false positives.

        Parameters
        ----------
        y_true : tf.Tensor
            The true segmentation masks.
        y_pred : tf.Tensor
            The predicted segmentation masks.
        weight_fn : float
            Weight for false negatives.
        weight_fp : float
            Weight for false positives.

        Returns
        -------
        loss : tf.Tensor
            The calculated weighted Dice loss.
        """

        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)

        # Ensure y_pred is between 0 and 1 (assuming this is not done in the model already)
        y_pred_f = tf.clip_by_value(y_pred_f, 0.0, 1.0)

        # Calculate the intersection (true positives)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)

        # Calculate the false negatives (FN) and false positives (FP)
        false_negatives = tf.reduce_sum(y_true_f * (1 - y_pred_f))  # Missed true regions
        false_positives = tf.reduce_sum((1 - y_true_f) * y_pred_f)  # Overpredicted regions

        # Apply weights to false negatives and false positives
        weighted_fn = weight_fn * false_negatives
        weighted_fp = weight_fp * false_positives

        numerator = 2 * intersection + tf.keras.backend.epsilon()  
        denominator = numerator + weighted_fn + weighted_fp + tf.keras.backend.epsilon() 

        dice_coefficient = numerator / denominator

        # Dice loss is 1 - Dice coefficient
        loss = 1 - dice_coefficient

        return loss

    
    def boundary_dice_loss(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        """
        Loss function that calculates
        overlap between ground truth 
        masks and predicted masks.

        Parameters
        ----------------
        y_true : tf.Tensor
            ground truth masks
        y_pred : tf.Tensor
            predicted mask

        Returns
        ------------
        dice_score : tf.Tensor
            Dice coefficient of
            predicted and ground
            truth masks
        """
    
        smooth = 1e-6
        sobel_filter = tf.image.sobel_edges
        y_true_edges = sobel_filter(y_true)
        y_pred_edges = sobel_filter(y_pred)

        y_true_f = tf.reshape(y_true_edges, [-1])
        y_pred_f = tf.reshape(y_pred_edges, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (\
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1 - tf.clip_by_value(dice_score, 0, 1)

        return dice_loss
    
    def save_generated_image(self,name,generated_image, epoch, step):
        image_array = generated_image.numpy().squeeze()

        image_array = (image_array * 255).astype(np.uint8)

        image = Image.fromarray(image_array)

        image.save(f"final_images_test/{name}_epoch_{epoch}_step_{step}.png")

    def create_csv_logger(self,log_path):
        # Create a CSV logger file and write the headers
        with open(log_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Step', 'Generator Loss', 'masker Loss','disc loss'])

    def update_csv_logger(self,log_path, epoch, step, gen_loss,mask_loss, disc_loss):
        # Append the new loss values to the CSV
        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, step, gen_loss.numpy(), mask_loss.numpy(),disc_loss.numpy()])

    def save_checkpoints(self,generator, masker,discriminator, epoch, step, checkpoint_dir="/data/pnlx/projects/mysell_masking_cnn/final_project/GAN_aug_test/checkpoints"):
        generator.save(f"{checkpoint_dir}/generator_full_aug_epoch_{epoch}_step_{step}.h5")
        masker.save(f"{checkpoint_dir}/masker_full_aug_epoch_{epoch}_step_{step}.h5")
        discriminator.save(f"{checkpoint_dir}/discriminator_full_aug_epoch_{epoch}_step_{step}.h5")

    def train(self, num_epochs : int = 100, batch_size : int = 4) -> None:
        """Start the training process for the generator and masker."""
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('path_to_save_model.h5', save_best_only=True)
        csv_logger = CSVLogger('training_log.csv', append=True, separator=',')

        log_path = 'training_log_gan_aug.csv'
        self.create_csv_logger(log_path)

        sub_data_dir = '/data/pnlx/projects/mysell_masking_cnn/training_set_refined_full_aug/'
        total_training_samples = 281600

        train_gen = self.training_generator(sub_data_dir, batch_size, 1, 101)
        val_gen = self.training_generator(sub_data_dir, batch_size, 101, 116)

        def normalize_generated_images(generated_images):
            normalized_images = np.empty_like(generated_images)
            
            for i in range(generated_images.shape[0]):  
                for j in range(generated_images.shape[-1]): 
                    sub_array = generated_images[i, :, :, j]
                    if np.sum(sub_array) > 0:
                        max_val = np.max(sub_array)
                        if max_val != 0: 
                            normalized_images[i, :, :, j] = sub_array / max_val
                        else:
                            normalized_images[i, :, :, j] = sub_array  
                    else:
                        normalized_images[i, :, :, j] = sub_array
            return normalized_images


        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for step, (mri_batch, mask_batch) in enumerate(train_gen):
                if step >= (total_training_samples / batch_size):  
                    break

                with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape, tf.GradientTape() as mask_tape:
                    generated_images = self.generator(mri_batch, training=True)
                    orig_generated_images = copy.deepcopy(generated_images)
                    #generated_images = normalize_generated_images(generated_images)
                    generated_images = tf.where(mask_batch > 0, mri_batch, generated_images)
                    mask_sums = tf.reduce_sum(mask_batch, axis=[1, 2])

                    # Create a tensor with 1s where the sum is greater than 0, and 0s otherwise
                    condition = mask_sums > 0
                    result_tensor = tf.where(condition, 1, 0)
                    zeros_tensor = tf.zeros_like(result_tensor)

                    disc_true = tf.concat([result_tensor, result_tensor,zeros_tensor], axis=0)
                    
                    disc_inp = tf.concat([mri_batch, generated_images,orig_generated_images], axis=0)

                    mask_inp = tf.concat([mri_batch, generated_images], axis=0)
                    mask_out = tf.concat([mask_batch, mask_batch], axis=0)

                    #disc_true = tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0)
                    permutation = tf.random.shuffle(tf.range(len(disc_true)))
                    disc_inp = tf.gather(disc_inp, permutation)
                    disc_true = tf.gather(disc_true, permutation)
                    mask_inp = tf.gather(mask_inp, permutation)
                    mask_out =tf.gather(mask_out, permutation)
                    disc_inp = disc_inp[:4]
                    disc_true = disc_true[:4]
                    mask_inp = mask_inp[:4]
                    mask_out = mask_out[:4]
                    disc_output = self.discriminator(disc_inp, training = True)
                    fake_output = self.masker(mask_inp, training=True)
                    mask_loss = self.masker_loss_fn(mask_out, fake_output)
                    disc_loss = self.discrimnator_loss_fn(disc_true, disc_output,mask_batch)

                    gen_loss = self.generator_loss_fn( mask_loss, mri_batch, generated_images,mask_batch, disc_loss)
                    gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                    self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

                    disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                    self.discriminator.optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

                    mask_gradients = mask_tape.gradient(mask_loss, self.masker.trainable_variables)
                    self.masker.optimizer.apply_gradients(zip(mask_gradients, self.masker.trainable_variables))

                if step % 100 == 0:
                    print(f"Step {step}, Generator Loss: {gen_loss.numpy()}, masker Loss: {mask_loss.numpy()}")
                    self.update_csv_logger(log_path, epoch + 1, step, gen_loss, mask_loss, disc_loss)
                    self.save_generated_image('generated_aug_full_aug',generated_images[0], epoch, step)
                    #self.save_generated_image('original',mri_batch[0], epoch, step)


if __name__ == '__main__':
    TrainModel().run_script()
