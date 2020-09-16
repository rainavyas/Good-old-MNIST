import pytorch


class Basic_two_layer(torch.nn.Module):
    def __init__(self, image_size, num_digits):

        super(Basic_single_layer, self).__init__()
        self.layer1 = torch.nn.Linear(2*num_digits, image_size, bias = True)
        self.layer1 = torch.nn.Linear(num_digits, 2*num_digits, bias = True)

    def forward(self, X):
        out1= self.layer1(X)
        out1_sig = torch.sigmoid(out1)
        y_unnormal = self.layer(out1_sig)
        y_SF = torch.nn.Softmax(y_unnormal, dim=1)
        #y_out = torch.argmax(y_SF, dim=1)
        return y_SF
