import torch
import torch.nn as nn

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=1):
        """
        Khởi tạo mô hình LSTM phát hiện té ngã.
        
        Args:
            input_size (int): Số lượng đặc trưng đầu vào mỗi frame (17 keypoints x 2 [x, y] = 34).
            hidden_size (int): Số lượng unit trong lớp ẩn của LSTM.
            num_layers (int): Số lượng lớp LSTM xếp chồng lên nhau.
            num_classes (int): Số lớp phân loại đầu ra (1 cho phân loại nhị phân Fall/Normal).
        """
        super(FallDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Lớp LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # Lớp Fully Connected phân loại
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Lan truyền tiến của mô hình.
        
        Args:
            x (torch.Tensor): Tensor đầu vào shape (batch_size, seq_len, input_size).
            
        Returns:
            out (torch.Tensor): Đầu ra shape (batch_size, num_classes).
        """
        # Khởi tạo trạng thái ẩn (hidden state) và trạng thái tế bào (cell state) với giá trị 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Chạy qua lớp LSTM
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Lấy đầu ra của frame cuối cùng trong chuỗi (last time step)
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        
        # Trả về kết quả (chưa qua sigmoid, sigmoid sẽ được tính trong loss function BCEWithLogitsLoss lúc train
        # hoặc tính thủ công khi inference)
        return out
