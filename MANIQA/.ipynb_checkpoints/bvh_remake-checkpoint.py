import os
import random
from decimal import Decimal
#整体矩阵添加高斯噪声
def gauss1(matrix,sigma):
    mu=0
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            matrix[i][j] += random.gauss(mu, sigma)
    return matrix

#特定行添加高斯噪声
def gauss2(matrix,sigma):
    mu=0
    for i in range(0, len(matrix)):
        matrix[i] += random.gauss(mu, sigma)
    return matrix

def All_fun(read_path,write_path,sigmad1):
    sum_out = []
    out_list=[]
    Flag = False

    f = open(read_path, "r")
    p = open(write_path,"w")
    #去除换行符
    readIn=f.readlines()
    readIn = [i.rstrip() for i in readIn]
    #查到到运动的motion数据
    for j in readIn:
        if Flag == True:
            out_list.append(j.split(" "))
          #  out_list.append(list(map(float,j)))
        if Flag==False:
            p.write(j)
            p.write('\n')
        if j.find("Frame Time:")>=0 and Flag==False:
            Flag = True
            continue

    #将读取的motion数据从string转换成float
    for i in range(len(out_list)):
        new_out = []
        for j in out_list[i]:
            new_out.append(float(j))
        sum_out.append(new_out)
    #整体矩阵添加高斯噪声
    sum_out=gauss1(sum_out,sigmad1)
    #处理好的数据写入到文件中
    for i in range(len(sum_out)):
        for j in range (len(sum_out[i])):
            k = str(Decimal(sum_out[i][j]).quantize(Decimal('0.000000')))
            p.write(k+' ')
        p.write('\n')
    f.close()
    p.close()

def Someone_fun(read_path,write_path,k,sigmad2): # k间隔多少帧开始进行加噪声

    sum_out = []
    out_list=[]
    Flag = False

    f = open(read_path, "r")
    p = open(write_path,"w")
    #去除换行符
    readIn=f.readlines()
    readIn = [i.rstrip() for i in readIn]
    #查到到运动的motion数据
    for j in readIn:
        if Flag == True:
            out_list.append(j.split(" "))
          #  out_list.append(list(map(float,j)))
        if Flag==False:
            p.write(j)
            p.write('\n')
        if j.find("Frame Time:")>=0 and Flag==False:
            Flag = True
            continue

    #将读取的motion数据从string转换成float
    for i in range(len(out_list)):
        new_out = []
        for j in out_list[i]:
            new_out.append(float(j))
        #达到特定行进行处理
        if i % k==0:
            new_out=gauss2(new_out,sigmad2)
        sum_out.append(new_out)

    #处理好的数据写入到文件中
    for i in range(len(sum_out)):
        for j in range (len(sum_out[i])):
            k = str(Decimal(sum_out[i][j]).quantize(Decimal('0.000000')))
            p.write(k+' ')
        p.write('\n')
    f.close()
    p.close()

def copy_path(read_path,write_path):
    file1 = open(read_path, "r")
    file2 = open(write_path, "w")
    s = file1.read()
    w = file2.write(s)
    file1.close()
    file2.close()
# (清空指定文件夹，不支持文件，文件夹不存在会报错)
def del_files2(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name)) # 删除一个空目录

if __name__ == "__main__":
    write_path = "E:/1.txt"
    read_path = "E:/aiming1_subject1.txt"
    folder_path = "/hy-tmp/lafan/"
    #new_path="E:/final_text/"
    motion=["/hy-tmp/final_test/motion_0/","/hy-tmp/final_test/motion_1/","/hy-tmp/final_test/motion_2/","/hy-tmp/final_test/motion_3/","/hy-tmp/final_test/motion_4/","/hy-tmp/final_test/motion_5/"]

    sigmad1 = 0.1  #整体矩阵添加高斯噪声
    sigmad2 = 0.1  #特定行添加高斯噪声

    k=10
    name_bvh=[]
    ori_path=[]
    full_path=[]
    #清空目标文件夹
    for i in range(0,6):
        del_files2(motion[i])
    #读取数据集中文件的名称

    for file_name in os.listdir(folder_path):
        #split的作用：以'.'为分界点，将test.xml分为两个部分，成为数组。[0]表示取前一部分
        name_bvh.append(file_name.split('.')[0])
    #print(name_bvh[76])
    #print(len(name_bvh))
    #创造出新的数据文件
    for i in range(0,len(name_bvh)):
        ori_path.append(folder_path + name_bvh[i] + '.bvh')
        for j in range(0,6):
        #    print(new_path + name_bvh[i]+'_'+str(j) + '.bvh')
            full_path.append(motion[j] + name_bvh[i]+'_'+str(j) + '.bvh')
        #    print(full_path[i*6+j])

    # 写入新的文件
    for i in range(len(full_path)):
        testFile = open(full_path[i], "w")
        testFile.close()
        yushu=i%6
        if yushu==0:
            #print(ori_path[i//6])
            #print(full_path[i])
            copy_path(ori_path[i//6], full_path[i])
        else :
            #Someone_fun(ori_path[i//6], full_path[i], (6-yushu)*k, sigmad2) #选取一些帧添加噪声
            All_fun(ori_path[i//6], full_path[i], yushu*sigmad1)        #全部添加噪声

