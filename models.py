import torch
import numpy as np

class EncoderResidualBlock(torch.nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_sz, downsample=True):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.downsample = downsample
        
        h_pad = int(kernel_sz / 2.0)
        
        self.conv_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_sz), padding=(0, h_pad), stride=(1,1)),
        )
        
        self.conv_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=(1, kernel_sz), padding=(0, h_pad), stride=(1,1)),   
        )
        
        # 1x1 Convolution with a horizontal stride of 2.
        # We need two of these because if in_ch < out_ch, we can't downsample both the output and the input (identity)
        # using the same layer.
        self.downsampler_id = torch.nn.Conv2d(in_ch, in_ch, kernel_size = (1, 1), padding=(0, 0), stride=(1,2))
        self.downsampler_out = torch.nn.Conv2d(out_ch, out_ch, kernel_size = (1, 1), padding=(0, 0), stride=(1,2))
        
        self.dropout = torch.nn.Dropout(p=0.1)
    
    def forward(self, X):
        
        # Store X - to be added to the output later as a skip connection.
        Id = X
        
        # Apply first convolution.
        X = self.conv_1(X)
        
        # Apply second convolution.
        X = self.conv_2(X)
    
        # Downsample.
        if self.downsample:
            X = self.downsampler_out(X)
            # we also need to downsample the original input so we can add the two.
            Id = self.downsampler_id(Id) 
        
        # Skip Connection.
        # If this is a residual block in which the number of channels increases,
        # add the input to the first in_ch dimensions of the output.
        # TODO: add twice if out_ch = 2 * in_ch.
        if self.in_ch < self.out_ch:
            pd = self.out_ch - self.in_ch
            X = X + torch.nn.functional.pad(Id, (0,0, 0,0, 0,pd, 0,0))
        else:
            X = X + Id
        
        # Apply Dropout.
        X = self.dropout(X)
        
        return X
            
class DecoderResidualBlock(torch.nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_sz, upsample=True):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        
        h_pad = int(kernel_sz / 2.0)
        
        self.conv_1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, kernel_sz), padding=(0, h_pad), stride=(1,1)),   
        )
        
        self.conv_2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(1, kernel_sz), padding=(0, h_pad), stride=(1,1)),
        )
        
        # 1x1 Convolution with a horizontal stride of 2.
        self.upsampler = torch.nn.ConvTranspose2d(out_ch, out_ch, kernel_size = (1, 1), padding=(0, 0), stride=(1,2), output_padding=(0,1))
        
        self.dropout = torch.nn.Dropout(p=0.1)
        
    def forward(self, X):
        
        # Apply first convolution.
        X = self.conv_1(X)
        
        # Apply second convolution.
        X = self.conv_2(X)
        
        # Upsample.
        if self.upsample:
            X = self.upsampler(X)
        
        # Apply Dropout.
        X = self.dropout(X)
        
        return X
        
class CNNAutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 1),
            EncoderResidualBlock(16, 16, 19, downsample=False),
            EncoderResidualBlock(16, 16, 19, downsample=True),
            EncoderResidualBlock(16, 32, 19, downsample=True),
            EncoderResidualBlock(32, 48, 19, downsample=True),
            EncoderResidualBlock(48, 64, 19, downsample=True),
            EncoderResidualBlock(64, 64, 19, downsample=True),
            EncoderResidualBlock(64, 80, 9, downsample=False),
            EncoderResidualBlock(80, 80, 9, downsample=False),
            EncoderResidualBlock(80, 80, 9, downsample=False)
        )

        self.decoder = torch.nn.Sequential(
            DecoderResidualBlock(80, 80, 9, upsample=False),
            DecoderResidualBlock(80, 80, 9, upsample=False),
            DecoderResidualBlock(80, 64, 9, upsample=False),
            DecoderResidualBlock(64, 64, 19, upsample=True),
            DecoderResidualBlock(64, 48, 19, upsample=True),
            DecoderResidualBlock(48, 32, 19, upsample=True),
            DecoderResidualBlock(32, 16, 19, upsample=True),
            DecoderResidualBlock(16, 16, 19, upsample=True),
            DecoderResidualBlock(16, 16, 19, upsample=False),
            torch.nn.ConvTranspose2d(16, 1, 1)
        )
        
    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
    
class CNNVariationalAutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        # Encoder.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 1),
            EncoderResidualBlock(16, 16, 19, downsample=False),
            EncoderResidualBlock(16, 16, 19, downsample=True),
            EncoderResidualBlock(16, 32, 19, downsample=True),
            EncoderResidualBlock(32, 48, 19, downsample=True),
            EncoderResidualBlock(48, 64, 19, downsample=True),
            EncoderResidualBlock(64, 64, 19, downsample=True),
            EncoderResidualBlock(64, 80, 9, downsample=False),
            EncoderResidualBlock(80, 80, 9, downsample=False),
            EncoderResidualBlock(80, 80, 9, downsample=False),
            torch.nn.Flatten()
        )

        # Decoder.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(60, 5120), # Upscale latent vector.
            torch.nn.Unflatten(1, (80, 1, 64)),
            DecoderResidualBlock(80, 80, 9, upsample=False),
            DecoderResidualBlock(80, 80, 9, upsample=False),
            DecoderResidualBlock(80, 64, 9, upsample=False),
            DecoderResidualBlock(64, 64, 19, upsample=True),
            DecoderResidualBlock(64, 48, 19, upsample=True),
            DecoderResidualBlock(48, 32, 19, upsample=True),
            DecoderResidualBlock(32, 16, 19, upsample=True),
            DecoderResidualBlock(16, 16, 19, upsample=True),
            DecoderResidualBlock(16, 16, 19, upsample=False),
            torch.nn.ConvTranspose2d(16, 1, 1)
        )
        
        # Layers for predicting parameters of q(z|x).
        self.latent_mu = torch.nn.Linear(5120, 60)
        self.latent_log_var = torch.nn.Linear(5120, 60)
    
    def sample_gaussian(self, mu, log_var):
        B_SZ = mu.shape[0]
        eps = torch.randn((B_SZ, 60))
        return mu + eps * torch.exp(log_var / 2.0) # multiply by std. dev.
    
    def forward(self, X):
        
        enc_out = self.encoder(X)
        
        # Predict parameters of q(z|x).
        z_mu = self.latent_mu(enc_out)
        z_log_var = self.latent_log_var(enc_out)
        
        # Reparameterization "trick". 
        # Get a multivariate gaussian with means z_mu and variances z_var * I.
        z = self.sample_gaussian(z_mu, z_log_var)
        
        # Pass latent sample through decoder.
        dec_out = self.decoder.forward(z)
        
        return dec_out, z_mu, z_log_var
    
    def sample(self, n):
        with torch.no_grad():
            z = torch.randn((n, 60))
            dec_out = self.decoder.forward(z)
            return dec_out
        
def vae_loss(X, dec_out, z_mu, z_log_var, alpha=1):
    
    # Reconstruction loss.
    #recl = torch.nn.functional.mse_loss(X, dec_out, reduction='mean')
    recl = torch.nn.functional.l1_loss(X, dec_out, reduction='mean')
    
    # KL Divergence.
    dkl = -0.5 * torch.sum(1 + z_log_var - torch.pow(z_mu, 2) - torch.exp(z_log_var),
                          axis=1)
    dkl = dkl.mean()
    
    return (recl + alpha * dkl), recl, dkl

class LSTMEncoder(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, X):
        enc_output, enc_hidden = self.lstm(X)
        return enc_output, enc_hidden
    
class LSTMDecoder(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        
        self.linear = torch.nn.Linear(hidden_size, input_size)
    
    def forward(self, X, enc_hidden):
        dec_output, dec_hidden = self.lstm(X, enc_hidden)
        dec_output = self.linear(dec_output.squeeze(0))
        return dec_output, dec_hidden
    
class LSTMAutoEncoder(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.decoder = LSTMDecoder(input_size, hidden_size)
    
    def forward(self, X, predict="recursive", tf_prob=0.0):
        
        # Use teacher-forcing randomly with probability tf_prob.
        if predict == "teacher-forcing" and torch.rand(1).item() < tf_prob:
            predict = "teacher-forcing"
        else:
            predict = "recursive"
        
        # Pass the input X (seq_len, batch_size, input_size) through the Encoder
        # and get its last hidden state.
        enc_output, enc_hidden = self.encoder(X)
        
        # Decoder gets its first hidden state from the encoder.
        hidden = enc_hidden
        
        # Decoder inputs.
        inputs = torch.zeros((1, X.shape[1], X.shape[2]))
        
        # If using teacher forcing, we pass the target (which is also X)
        # as the input at each time-step.
        if predict == "teacher-forcing":
            inputs = X
        
        # Decoder outputs.
        outputs = torch.zeros((0, X.shape[1], X.shape[2]))
        
        n_steps = X.shape[0]
        for t in range(n_steps):
            
            # The input to the decoder at the current time-step is inputs[t:t+1].
            # We preserve the first dimension, because nn.LSTM expects the input 
            # to have shape (L, N, D).
            
            # Get next decoder output and hidden state.
            output, hidden = self.decoder(inputs[t:t+1], hidden)
            output = torch.unsqueeze(output, dim=0) # (N, D) => (1, N, D).
            
            # Append the output to the outputs tensor.
            outputs = torch.cat((outputs, output), dim=0)
            
            # Append the output to the inputs tensor, 
            # since the input at time t+1 is the current output.
            if predict == "recursive":
                inputs = torch.cat((inputs, output), dim=0)
            
        return outputs

class StackedLSTM(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden):
        super().__init__()
        layers = []
        layers.append(torch.nn.LSTM(input_dim, hidden_dim))
        for _ in range(n_hidden):
            layers.append(torch.nn.LSTM(hidden_dim, hidden_dim))
        layers.append(torch.nn.LSTM(hidden_dim, output_dim))
        self.layers = layers
        
    def forward(self, x):
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            x, (hn, cn) = layer(x)
            if i < n_layers - 1:
                x = torch.nn.functional.relu(x)
        return x, hn.squeeze()
    
class StackedLSTMAutoEncoder(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_hidden):
        super().__init__()
        self.enc = StackedLSTM(input_dim, hidden_dim, latent_dim, n_hidden)
        self.dec = StackedLSTM(latent_dim, hidden_dim, hidden_dim, n_hidden)
        self.lin = torch.nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        _, z = self.enc(x)
        z = z.repeat((x.shape[0], 1, 1))
        x, _ = self.dec(z)
        return self.lin(x)