"""
克麦波吉2025/6/4分享

LBM计算点声源干涉
自己写了点后处理生成彩色的图片
如果各位佬有用到的这个代码的，还望不吝美言
然后如果有HR看到的，希望能给个面试机会，谢谢啦

个人QQ:2913317255
   微信：zl13297143985
"""

import taichi as ti
import tifffile
import numpy as np
import math
ti.init(arch=ti.gpu)  # 优先GPU
from PIL import Image
# D2Q9模型
D = 2
Q = 9

# 仿真的范围
NX,NY = 1280,720

# 幅值
amplitude = 0.1

# 频率
frequency = 7

# ----参数以及物理量----
tau = 0.51           # 决定单组分粘度

print("仿真范围：",NX,NY)

omega = ti.field(ti.f32, shape=Q)   # 每个方向上的权重
ecx = ti.field(ti.i32, shape=Q)     # 格子速度
ecy = ti.field(ti.i32, shape=Q)     # 格子速度
opp = ti.field(ti.i32, shape=Q)     # 离散速度的相反序号

old_omega = [4.0 / 9.0,
             1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
             1.0 / 36.0, 1.0 / 36.0,1.0 / 36.0, 1.0 / 36.0]
old_ecx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
old_ecy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
old_opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# 初始filed
for i in range(Q):
    omega[i] = old_omega[i]
    ecx[i] = old_ecx[i]
    ecy[i] = old_ecy[i]
    opp[i] = old_opp[i]

# 组分的物理量
rho = ti.field(ti.f32, shape=(NX, NY))  # 组分的密度
u = ti.field(ti.f32, shape=(NX, NY))  # 组分的x分量速度
v = ti.field(ti.f32, shape=(NX, NY))  # 组分的y分量速度
boundary_types = ti.field(ti.f32, shape=(NX, NY))   # 固体边界，可以通过在初始化更改这个field来实现不同的固体边界

# 输出图片的RGB向量场
out_png_field = ti.Vector.field(n=3, dtype=int, shape=(NX, NY))

# 分布函数
fin = ti.field(ti.f32, shape=(NX, NY, Q))  # 第一组分布函数
fout = ti.field(ti.f32, shape=(NX, NY, Q))  # 第二组分布函数

@ti.func
def feq(u_f: float, v_f: float, rho_f: float, q_f: int) -> float:
    """
    计算得到Feq（分布函数非平衡部分）
    :param u_f: 速度u
    :param v_f: 速度v
    :param w_f: 速度w
    :param rho_f: 密度
    :param q_f: 离散的速度方向
    :return: Feq
    """
    eu = u_f * ecx[q_f] + v_f * ecy[q_f]
    u2 = u_f * u_f + v_f * v_f
    return omega[q_f] * rho_f * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)

@ti.kernel
def init():
    """
    初始化整个场
    :return:
    """
    for i, j in ti.ndrange(NX, NY):
        if i == 0 or i == NX or j == 0 or j == NY:
            boundary_types[i, j] = 1
        else:
            boundary_types[i, j] = 0

        if i == 720 and (j<357 or j > 363):
            boundary_types[i, j] = 1
        # 流体域初始化
        if boundary_types[i, j] == 0:
            rho[i, j] = 1.0
        else:
            rho[i, j] = 0.000001
        u[i, j] = 0.0
        v[i, j] = 0.0
        for q in ti.ndrange(Q):
            fin[i, j,q] = feq(u[i, j],v[i, j],rho[i, j],q)


@ti.kernel
def collision():
    """
    碰撞步
    :return:
    """
    for i, j, q in ti.ndrange((1, NX - 1), (1, NY - 1), Q):
        if boundary_types[i, j] == 0:
            fout[i, j,  q] = fin[i, j,  q] - (fin[i, j,  q] - feq(u[i, j], v[i, j], rho[i, j], q)) / tau


@ti.kernel
def boundary():
    """
    边界
    :return:
    """
    for i, j in ti.ndrange((1, NX - 1), (1, NY - 1)):
        # 如果是固体
        if boundary_types[i, j] == 1:
            for q in ti.ndrange(Q):
                fout[i, j, q] = fout[i+ecx[q], j+ecy[q], opp[q]]

    for i in ti.ndrange((1,NX-1)):
        fout[i, 0, 2] = fout[i, 1, 4]
        fout[i, 0, 5] = fout[i+1, 1, 7]
        fout[i, 0, 6] = fout[i-1, 1, 8]
        fout[i, NY-1, 4] = fout[i, NY-2, 2]
        fout[i, NY-1, 7] = fout[i-1, NY-2, 5]
        fout[i, NY-1, 8] = fout[i+1, NY-2, 6]

    for j in ti.ndrange((1, NY - 1)):
        fout[0, j, 1] = fout[1, j, 3]
        fout[0, j, 5] = fout[1, j+1, 7]
        fout[0, j, 8] = fout[1, j-1, 6]
        fout[NX-1, j, 3] = fout[NX-2, j, 1]
        fout[NX-1, j, 6] = fout[NX-2, j+1, 8]
        fout[NX-1, j, 7] = fout[NX-2, j-1, 5]

    # 四个顶点
    fout[0, 0, 5] = fout[1, 1, 7]
    fout[0, NY - 1, 8] = fout[1, NY - 2, 6]
    fout[NX - 1, 0, 6] = fout[NX - 2, 1, 8]
    fout[NX - 1, NY - 1, 7] = fout[NX - 2, NY - 2, 5]


@ti.kernel
def stream():
    """
    迁移
    如果需要周期性边界，可以把周期性边界添加在迁移步，减少运算
    :return:无返回参数
    """
    for i, j in ti.ndrange((1, NX - 1), (1, NY - 1)):
        # # 非固体域执行迁移
        if boundary_types[i, j] == 0:
            for q in ti.ndrange(Q):
                fin[i, j, q] = fout[i - ecx[q], j - ecy[q], q]

@ti.kernel
def statistics():
    """
    宏观统计
    :return:
    """
    for i, j in ti.ndrange((1, NX - 1), (1, NY - 1)):
        # 流体域
        if boundary_types[i, j] == 0:
            # 统计更新密度，速度
            rho[i, j] = 0.0
            u[i, j] = 0.0
            v[i, j] = 0.0
            for q in ti.ndrange(Q):
                rho[i, j] += fin[i, j, q]
                u[i, j] += fin[i, j, q] * ecx[q]
                v[i, j] += fin[i, j, q] * ecy[q]
            u[i, j] = u[i, j] / rho[i, j]
            v[i, j] = v[i, j] / rho[i, j]



@ti.func
def float_to_rgb(in_float:ti.f32,in_min:ti.f32,in_max:ti.f32):
    """
    后处理的可视化的color map,类似的可以设置别的colormap
    :param in_float: 待转化的输入数据
    :param in_min: colormap显示的下限
    :param in_max: colormap显示的上限
    :return:
    """

    cha = (in_max - in_min)/6.0
    r = 0.0
    g = 0.0
    b = 0.0
    if in_float < in_min:
        r = 255
        g = 0
        b = 0
    # 1红--橙
    elif in_min <=  in_float <= in_min+cha:
        r = 255
        g = 152*(in_float-in_min)/cha
        b = 0
    # 2橙--黄
    elif in_min+cha <= in_float <= in_min + 2.0*cha:
        r = 255
        g = 152 + 103*(in_float - in_min - cha) / cha
        b = 0
    # 3黄--绿
    elif in_min+2.0*cha <= in_float <= in_min + 3.0*cha:
        r = 255 - 255*(in_float - in_min - 2.0*cha) / cha
        g = 255
        b = 0
    # 4绿--青
    elif in_min+3.0*cha <= in_float <= in_min + 4.0*cha:
        r = 0
        g = 255
        b = 255*(in_float - in_min-3.0*cha)/cha
    # 5青--蓝
    elif in_min+4.0*cha <= in_float <= in_min + 5.0*cha:
        r = 0
        g = 255-(in_float - in_min-4.0*cha)/cha
        b = 255
    # 6蓝--紫
    elif in_min+5.0*cha <= in_float <= in_min + 6.0*cha:
        r = 150*(in_float - in_min-5.0*cha) / cha
        g = 0
        b = 255

    elif in_float > in_max:
        r = 150
        g = 0
        b = 255
    # # 确保数据不超过范围
    # if r < 0:
    #     r = 0
    # if r >255:
    #     r = 255
    #
    # if g < 0:
    #     g = 0
    # if g >255:
    #     g = 255
    #
    # if b < 0:
    #     b = 0
    # if b >255:
    #     b = 255
    return int(r),int(g),int(b)

@ti.kernel
def RGB_PNG():
    """
    填充图片的RGB向量场
    :return:
    """
    for i, j in ti.ndrange(NX,NY):
        out_png_field[i, j] = float_to_rgb(rho[i, j],0.998,1.002)


def out_png(filename):
    """
    输出RGB的PNG
    :param filename:
    :return:
    """
    png_np = out_png_field.to_numpy()
    img_data = png_np.astype(np.uint8)
    if img_data.shape[0] == NX and img_data.shape[1] == NY:
        # 转置为 (高度, 宽度, 通道)
        img_data = img_data.transpose(1, 0, 2)
    # 创建图像对象
    img = Image.fromarray(img_data, 'RGB')
    img.save(filename)


# @ti.func
# @ti.kernel
# 目前似乎无法把这一部分代码放到ti.kernel内核或者ti.func中，所以输出文件这一操作可能会比较占用时间
def out_tiff(in_iter):
    """
    直接输出tiff文件
    :param in_iter: 步数
    :return:无返回值，仅生成tiff文件
    """

    rho_arr = rho.to_numpy()
    rho_arr = rho_arr[1:NX - 1, 1:NY - 1]
    tifffile.imwrite('OutPut/' + str(in_iter) + 'rho.tiff', rho_arr)  # 将numpy数组切片后输出tiff文件

    u_arr = u.to_numpy()
    u_arr = u_arr[1:NX - 1, 1:NY - 1]
    tifffile.imwrite('OutPut/' + str(in_iter) + 'u.tiff', u_arr)  # 将numpy数组切片后输出tiff文件

    v_arr = v.to_numpy()
    v_arr = v_arr[1:NX - 1, 1:NY - 1]
    tifffile.imwrite('OutPut/' + str(in_iter) + 'v.tiff', v_arr)  # 将numpy数组切片后输出tiff文件

def lbm_main():
    """
    LBM计算的主程序
    :return: 无返回值
    """

    print("----Start initialization----")
    init()          # 初始化
    print("----End of initialization----")

    # 计算总循环
    print("----Start the calculation----")
    for iter in range(1501):
        rho_in = 1.0 + amplitude*math.sin(iter/frequency)
        # evolution（求解） 碰撞->更新边界->迁移->宏观统计更新->更新实际速度->计算判断->选择数据输出
        collision()
        boundary()
        stream()
        statistics()
        # 波动源
        rho[360,300] = rho_in
        rho[360, 420] = rho_in
        if iter % 100 == 0:
            print("step:", iter)
            out_tiff(iter)

        RGB_PNG()
        out_png('OutPut_PNG/' + str(iter)+'.png')
    # RGB_PNG()
    # out_png('OutPut_PNG/' + "0000" + '.png')
    # print(rho[10, 10])
    # print(out_png_field[10, 10])


if __name__ == '__main__':
    lbm_main()
