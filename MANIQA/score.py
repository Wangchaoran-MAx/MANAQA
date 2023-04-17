import os
write_path ='/hy-tmp/score.txt'
folder_path = "/hy-tmp/all_motion"
name_bvh=[]
# 读取数据集中文件的名称
for file_name in os.listdir(folder_path):
    name_bvh.append(file_name)
#print(name_bvh)
f=open(write_path,'w')
for i in range(len(name_bvh)):
    yushu=i%6
    a=5-yushu
    f.write(name_bvh[i]+' '+str(a)+'\n')
f.close()