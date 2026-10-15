import torch


def main():
    # 固定 Random Seed，保证结果可复现。
    torch.manual_seed(42)

    # 原始二维数据点 x0。
    x0 = torch.tensor([2.0, 1.0])

    # 定义 Linear Beta Schedule 和累计乘积 alpha_bar。
    T = 100
    beta = torch.linspace(0.0001, 0.02, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 通过 Closed-form Sampling，直接采样各时间步的 x_t。
    # 索引从 0 开始：t=0 已进行第一步加噪，并非原始 x0。
    for t in [0, 24, 49, 74, 99]:
        epsilon = torch.randn_like(x0)
        x_t = (
            torch.sqrt(alpha_bar[t]) * x0
            + torch.sqrt(1 - alpha_bar[t]) * epsilon
        )
        print(f"t={t:2d}, alpha_bar[t]={alpha_bar[t].item():.6f}, x_t={x_t}")


if __name__ == "__main__":
    main()
