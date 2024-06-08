import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import torch
import os

def normalize_loss(values, new_min=0, new_max=1):
    old_min = np.min(values)
    old_max = np.max(values)
    return new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)

def train(model, train_loader, criterion, optimizer, epochs, device, loss_file):
    model.train()
    all_losses = []
    l2_error_u_epoch = []
    l2_error_v_epoch = []
    l2_error_p_epoch = []
    l2_error_total_epoch = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        u_error_epoch = 0.0
        v_error_epoch = 0.0
        p_error_epoch = 0.0
        total_error_epoch = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 假设outputs和labels是(batch_size, num_features)
            outputs = outputs.view(-1, 3)  # 调整维度到 (batch_size, 3)
            labels = labels.view(-1, 3)  # 调整维度到 (batch_size, 3)

            # Calculate L2 errors for U, V, P, and Total
            u_error = torch.sqrt(torch.sum((outputs[:, 0] - labels[:, 0]) ** 2)) / torch.sqrt(torch.sum(labels[:, 0] ** 2))
            v_error = torch.sqrt(torch.sum((outputs[:, 1] - labels[:, 1]) ** 2)) / torch.sqrt(torch.sum(labels[:, 1] ** 2))
            p_error = torch.sqrt(torch.sum((outputs[:, 2] - labels[:, 2]) ** 2)) / torch.sqrt(torch.sum(labels[:, 2] ** 2))
            total_error = torch.sqrt(torch.sum((outputs - labels) ** 2)) / torch.sqrt(torch.sum(labels ** 2))

            u_error_epoch += u_error.item()
            v_error_epoch += v_error.item()
            p_error_epoch += p_error.item()
            total_error_epoch += total_error.item()

        avg_loss = running_loss / len(train_loader)
        avg_u_error = u_error_epoch / len(train_loader)
        avg_v_error = v_error_epoch / len(train_loader)
        avg_p_error = p_error_epoch / len(train_loader)
        avg_total_error = total_error_epoch / len(train_loader)

        all_losses.append(avg_loss)
        l2_error_u_epoch.append(avg_u_error)
        l2_error_v_epoch.append(avg_v_error)
        l2_error_p_epoch.append(avg_p_error)
        l2_error_total_epoch.append(avg_total_error)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Normalize all errors
    l2_error_u = normalize_loss(np.array(l2_error_u_epoch))
    l2_error_v = normalize_loss(np.array(l2_error_v_epoch))
    l2_error_p = normalize_loss(np.array(l2_error_p_epoch))
    l2_error_total = normalize_loss(np.array(l2_error_total_epoch))

    # 保存 loss 值到文件
    np.savez(loss_file, loss=all_losses, l2_error_u=l2_error_u, l2_error_v=l2_error_v, l2_error_p=l2_error_p, l2_error_total=l2_error_total)

    return np.array(all_losses), l2_error_u, l2_error_v, l2_error_p, l2_error_total

def evaluate(model, test_inputs, test_labels, device):
    model.eval()
    with torch.no_grad():
        test_inputs, test_labels = torch.tensor(test_inputs, dtype=torch.float32).to(device), torch.tensor(test_labels, dtype=torch.float32).to(device)
        predictions = model(test_inputs).cpu().numpy()
        l2_error = np.sqrt(np.sum((predictions - test_labels.cpu().numpy()) ** 2)) / np.sqrt(np.sum(test_labels.cpu().numpy() ** 2))
        print(f"L2 Error: {l2_error}")

def visualize(model, test_inputs, test_labels, losses, path="./videos", device="cpu", dpi=300):
    model.eval()
    test_inputs, test_labels = torch.tensor(test_inputs, dtype=torch.float32).to(device), torch.tensor(test_labels, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(test_inputs).cpu().numpy()
    
    sample_t, sample_x, sample_y, _ = test_inputs.shape

    u_vmin, u_vmax = np.percentile(test_labels[:, :, :, 0].cpu(), [0.5, 99.5])
    v_vmin, v_vmax = np.percentile(test_labels[:, :, :, 1].cpu(), [0.5, 99.5])
    p_vmin, p_vmax = np.percentile(test_labels[:, :, :, 2].cpu(), [0.5, 99.5])

    vmin_list = [u_vmin, v_vmin, p_vmin]
    vmax_list = [u_vmax, v_vmax, p_vmax]

    output_names = ["U", "V", "P"]
    if not os.path.isdir(path):
        os.makedirs(path)

    t_set = []
    if sample_t < 100:
        t_set = np.arange(sample_t, dtype=np.int32)
    else:
        for t in range(sample_t):
            if t % int(sample_t / 50) == 0 or t == sample_t - 1:
                t_set.append(t)

    l2_error_u = []
    l2_error_v = []
    l2_error_p = []
    l2_error_total = []

    for t in t_set:
        u_label = test_labels[t, :, :, 0].cpu().numpy()
        v_label = test_labels[t, :, :, 1].cpu().numpy()
        p_label = test_labels[t, :, :, 2].cpu().numpy()

        u_predict = predictions[t, :, :, 0]
        v_predict = predictions[t, :, :, 1]
        p_predict = predictions[t, :, :, 2]

        u_label_2d = np.reshape(np.array(u_label), (sample_x, sample_y))
        v_label_2d = np.reshape(np.array(v_label), (sample_x, sample_y))
        p_label_2d = np.reshape(np.array(p_label), (sample_x, sample_y))

        u_predict_2d = np.reshape(np.array(u_predict), (sample_x, sample_y))
        v_predict_2d = np.reshape(np.array(v_predict), (sample_x, sample_y))
        p_predict_2d = np.reshape(np.array(p_predict), (sample_x, sample_y))

        u_error_2d = np.abs(u_predict_2d - u_label_2d)
        v_error_2d = np.abs(v_predict_2d - v_label_2d)
        p_error_2d = np.abs(p_predict_2d - p_label_2d)

        label_2d = [u_label_2d, v_label_2d, p_label_2d]
        predict_2d = [u_predict_2d, v_predict_2d, p_predict_2d]
        error_2d = [u_error_2d, v_error_2d, p_error_2d]

        lpe_2d = [label_2d, predict_2d, error_2d]
        lpe_names = ["label", "predict", "error"]

        fig = plt.figure(dpi=dpi)

        gs = gridspec.GridSpec(4, 3)  # 增加一个行用于显示误差变化曲线

        title = "t={:d}".format(t)
        plt.suptitle(title, fontsize=14)

        gs_idx = int(0)

        for i, data_2d in enumerate(lpe_2d):
            for j, data in enumerate(data_2d):
                ax = fig.add_subplot(gs[gs_idx])
                gs_idx += 1

                if lpe_names[i] == "error":
                    img = ax.imshow(data.T, vmin=0, vmax=1, cmap=plt.get_cmap("jet"), origin='lower')
                else:
                    img = ax.imshow(data.T, vmin=vmin_list[j], vmax=vmax_list[j], cmap=plt.get_cmap("jet"), origin='lower')

                ax.set_title(output_names[j] + " " + lpe_names[i], fontsize=4)
                plt.xticks(size=4)
                plt.yticks(size=4)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img, cax=cax)

        gs.tight_layout(fig, pad=0.4, w_pad=0.4, h_pad=0.4)
        plt.savefig(os.path.join(path, f"FlowField_{t+1}.png"))
        plt.close(fig)

        # 计算误差
        error = np.abs(u_error_2d) + np.abs(v_error_2d) + np.abs(p_error_2d)
        l2_error_u.append(np.sqrt(np.sum(u_error_2d ** 2)) / np.sqrt(np.sum(u_label_2d ** 2)))
        l2_error_v.append(np.sqrt(np.sum(v_error_2d ** 2)) / np.sqrt(np.sum(v_label_2d ** 2)))
        l2_error_p.append(np.sqrt(np.sum(p_error_2d ** 2)) / np.sqrt(np.sum(p_label_2d ** 2)))
        l2_error_total.append(np.sqrt(np.sum(error ** 2)) / np.sqrt(np.sum(label_2d[0] ** 2)))

    # 绘制误差变化曲线
    # fig, ax = plt.subplots(dpi=dpi)
    # t_set = np.array(t_set)
    # ax.plot(t_set, l2_error_u, 'b--', label="l2_error of U")
    # ax.plot(t_set, l2_error_v, 'g-.', label="l2_error of V")
    # ax.plot(t_set, l2_error_p, 'k:', label="l2_error of P")
    # ax.plot(t_set, l2_error_total, 'r-', label="l2_error of All")
    # ax.legend()
    # ax.set_xlabel('time')
    # ax.set_ylabel('l2_error')
    # ax.set_xticks(np.arange(0, len(t_set), step=5))
    # ax.set_ylim(0, 0.5)  # 设置纵向轴的范围
    # ax.set_yticks(np.arange(0, 0.5, step=0.05))  # 设置纵向轴的刻度步长为0.05
    # plt.savefig(os.path.join(path, "Error_Curve.png"))
    # plt.close(fig)

    # 归一化loss值
    normalized_losses = normalize_loss(losses)
    fig, ax = plt.subplots(dpi=dpi)
    epochs = np.arange(len(normalized_losses))
    ax.plot(epochs, normalized_losses, 'b-', label="Normalized Loss")
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Loss')
    ax.set_ylim(0, 1)
    plt.savefig(os.path.join(path, "Normalized_Loss.png"))
    plt.close(fig)