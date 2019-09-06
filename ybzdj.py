from matplotlib import pyplot as plt
#元胞自动机、流言模型
 
 
def count_rumour(matrix,rumour):#二维矩阵,计算流言的数目
    sum_rumour=0
    for sublist in matrix:
        sum_rumour+=sublist.count([0,rumour])+sublist.count([rumour,rumour])
    return sum_rumour
 
def spread(size,rumour,start_x,start_y):
    # 初始化
    rumour_matrix = [[[0, 0] for i in range(size)] for j in range(size)]
    rumour_spread = []
    rumour_matrix[start_x][start_y] = [1, 1]
    rumour_spread.append(count_rumour(rumour_matrix, rumour))
    # 个体更新
    while count_rumour(rumour_matrix, rumour) < size * size:
        for i in range(size):
            for j in range(size):
                # 制定流言传播规则1、准备传播流言的传播给邻居（上下左右）2、上次听到流言的，变为准备传播流言
                if rumour_matrix[i][j][0] == rumour:
                    if i - 1 >= 0:
                        rumour_matrix[i - 1][j] = [rumour,rumour]#已经遍历了
                    if i + 1 < size:
                        rumour_matrix[i + 1][j][1] = rumour
                    if j - 1 >= 0:
                        rumour_matrix[i][j - 1] = [rumour,rumour]#已经遍历了
                    if j + 1 < size:
                        rumour_matrix[i][j + 1][1] = rumour
                elif rumour_matrix[i][j][1] == rumour:
                    rumour_matrix[i][j][0] = rumour
        rumour_spread.append(count_rumour(rumour_matrix, rumour))
    print(rumour_spread[:10])#打印前十个时间步流言传播的速度
    plt.plot(rumour_spread)
 
 
 
#设置参数
size=200
rumour=1
#最里面的列表中第2个元素为rumour说明更新刚听到流言，第1个元素为rumour说明准备传播流言
center_x,center_y=int(size/2),int(size/2)
spread(size,rumour,center_x,center_y)
spread(size,rumour,0,0)
plt.show()