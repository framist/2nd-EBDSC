import torch

if __name__ == '__main__':
    from time import time
    # from models.TimesNet import Model, Configs
    from models.iTransformer import Model, Configs
    

    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    # input = torch.rand(10, 1024, 5, 128).cuda()
    input = torch.rand(10, 1024, 5).cuda()
    # model = Model(wide_value_emb=True).cuda()
    model = Model(wide_value_emb=False).cuda()
    
    print("模型参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    start = time()
    pred_series = model(input)
    end = time()
    print(pred_series.shape, f"time {end - start}")
    # print(model)
