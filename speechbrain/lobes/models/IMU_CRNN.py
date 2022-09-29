import os
import torch
import torch.nn as nn
import torch.nn.functional as F



def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))


class IMU_CRNN_GRU(nn.Module): # IMU_CRNN_Ott_GRU_3
    '''
    Modified net from Ott 2022
    GRU 2 with fewer neurons
    '''

    def __init__(self, dropout_cnn=0.5, dropout_rnn=0.2, rnn_width=60):
        super().__init__()

        channel_num_1 = 128
        channel_num_2 = 200

        self.down = nn.AvgPool1d(kernel_size=10, stride=5, padding=4)

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=channel_num_1, kernel_size=3, stride=1, padding=1)  # floor(500 + 2*p - 3 + 1) = 500
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.norm1 = nn.BatchNorm1d(num_features=channel_num_1)
        self.drop1 = nn.Dropout(p=dropout_cnn)

        self.conv2 = nn.Conv1d(in_channels=channel_num_1, out_channels=channel_num_2, kernel_size=3, stride=1, padding=1)  # (250 + 2*2 - 4) / 1 = 250
        self.norm2 = nn.BatchNorm1d(num_features=channel_num_2)
        self.drop2 = nn.Dropout(p=dropout_cnn)  # [B, C2, T]

        self.rnn = nn.GRU(input_size=channel_num_2, hidden_size=rnn_width, num_layers=2,
                           bias=True, batch_first=True, dropout=dropout_rnn, bidirectional=True)
        self.drop3 = nn.Dropout(p=dropout_rnn)

        self.fc = nn.Linear(in_features=rnn_width*2, out_features=1)

    def forward(self, x, cls=True):
        '''
        If don't want classification output, set cls=False
        '''
        if ('CUDA_VISIBLE_DEVICES' not in os.environ) or len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            self.rnn.flatten_parameters()
        x = self.down(x)  # [B, 64, 500]

        x = F.relu_(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.drop1(x)  # [B, C1=200, T=50]

        x = F.relu_(self.conv2(x))
        x = self.norm2(x)
        x = self.drop2(x)  # [B, C2=200, T=25]

        x = x.permute([0, 2, 1])  # [B, T=25, C2=256]
        x, _ = self.rnn(x)  # [B, T=25, 512]
        x = self.drop3(x) # [B, T, F=120]

        if cls==True:
            x = torch.sigmoid(self.fc(x))
            x = x.squeeze()
        else:
            pass

        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, input):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        return x


# if __name__ == '__main__':
#     main()
