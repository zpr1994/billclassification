# billclassification 利用CNN实现票据分类

## 依赖库
python 3.5 <br>
pytorch 0.3.0 <br>
matplotlib <br>
numpy <br>

## 数据集
票据图像总共五类：车票、定额发票、机打发票、机打小票和财务报销单，共765张。 <br>
下载链接：未完成 <br>

## 标注
标注文件位于 billcalssificaiton/label <br>
格式为 XXXX/XXX.jpg N.数据共5类 所以N取0-4. <br>
未划分训练集和测试集 <br>

## 训练
提供三种训练方法：1、100次随机留出；2、10次10折交叉检验；3、单次训练。 <br>
1、2的训练、测试数据皆由标注文件生成，会保存每次实验中最高准确率的结果可用做3的输入。 <br>

## 结果
基于交叉检验中最好结果--准确率折线图 <br>
![](https://github.com/zpr1994/billclassification/raw/master/plt/acc.jpg)

基于交叉检验中最好结果--损失折线图 <br>
![](https://github.com/zpr1994/billclassification/raw/master/plt/loss.jpg)

基于交叉检验中最好结果--5类准确率柱状图 <br>
![](https://github.com/zpr1994/billclassification/raw/master/plt/simple_bar.jpg)
