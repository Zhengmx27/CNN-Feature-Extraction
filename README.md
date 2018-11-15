# CNN-Feature-Extraction
CNN 用来对向量进行特征提取，向量可以是文本的embedding、社交网络节点的embedding、图片等
这里用Cora 这个数据集的embedding做训练集和测试集。。。
cnn90_N2V_W100.txt ：训练数据集
Cora_category.txt ：训练数据集标签
else_data_80%.txt ：测试数据集
else_data_80%_vec.txt ： 测试集标签
cnn.py : 进行简单的卷积操作 然后进行特征提取


卷积操作 这里只是做了一个示范，可以根据自己的实际情况进行修改层数等，这里特征提取主要参数不是后面的准确率
是后面的 feature = sess.run(h_fc1, feed_dict={x: test_x}) 
h_fc1 是全连接层的输出，即提取的特征。

reference：
https://www.cnblogs.com/yangmang/p/7528935.html
https://www.cnblogs.com/chuxiuhong/p/6132814.html
