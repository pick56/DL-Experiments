# 说明


原先的convolutional文件是大佬提供的，依靠命令行运行主要有几个选择
* 可以选择浮点数的数据格式
* 可以进行自我测试

但是在实际的使用中发现，16位的浮点数会导致一些运行时无法结束，自我测试
其实只是在测试的时候检测代码有没有写错，fakedata函数也没有特殊含义
所以后面在helloworld文件中进行了重构，去掉了这些冗余的部分。
