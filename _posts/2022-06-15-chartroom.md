---
title: JAVA-Socket通信 聊天室 
date: 2022-06-15 21:02:00 +0800
categories: [课设]
tags: [Java]
pin: true
author: Aye486

toc: true
comments: true
typora-root-url: ../../Aye486.github.io
math: false
mermaid: true
---

聊天室曾经盛行一时，今天我们就用简单的java代码来复刻他。

## 一、项目名称

聊天室

## 二、功能介绍

1. 用Java图形用户界面编写聊天室服务器端和客户端， 支持多个客户端连接到一个服务器。每个客户端能够输入账号，**包括注册功能**。

2. 可以实现群聊（聊天记录显示在所有客户端界面）。

3. 完成好友列表在各个客户端上显示，**包括头像和用户名**。

4. 可以实现私人聊天，用户可以选择某个其他用户，单独发送信息，同时实现了文件传输。

5. 服务器能够群发系统消息，**能够对用户私发消息**，能够强行让某些用户下线。

6. 客户端的上线下线要求能够在其他客户端上面实时刷新。
7. **服务器能够查看在线用户和注册用户**

## 三、模块功能

### 服务器与客户端之间的交互简图

![无标题](/assets/blog_res/2022-06-15-chartroom.assets/%E6%97%A0%E6%A0%87%E9%A2%98.png)

### 服务器与客户端的功能缩略图

![image-20220617114406232](/assets/blog_res/2022-06-15-chartroom.assets/image-20220617114406232.png)

![image-20220617114451101](/assets/blog_res/2022-06-15-chartroom.assets/image-20220617114451101.png)

### 具体框架

#### Client端

![image-20220617112313989](/assets/blog_res/2022-06-15-chartroom.assets/image-20220617112313989.png)

#### Server端

![image-20220617112331772](/assets/blog_res/2022-06-15-chartroom.assets/image-20220617112331772.png)

#### 工具类

![image-20220617112346610](/assets/blog_res/2022-06-15-chartroom.assets/image-20220617112346610.png)

<!--具体功能其查看其他相关博客-->

## 四、开发模式：MVC模式

M:model(数据层)

V：view(显示层)

C：controller(控制层)

## 五：以上是大概的框架

### UML图（大致构思）：

![image-20220616224447338](/assets/blog_res/2022-06-15-chartroom.assets/image-20220616224447338.png)

## 总结

在网络越来越发达的今天, 人们越来越依赖于网络, 越来越离不开网络, 由此而产生的聊天工具越来越多, 类似QQ、MSN, 网络聊天一类的聊天系统发展日新月异。为了更好充实人们的生活, 特做此系统满足人们在日常生活的需要和需求, 也为了满足人们在信息流通方面的方便, 使人们更能分享互联网上的资源, 使网络的意义更能充分体现。经过了长时间的学习，终于完成了本次课设，本聊天室是使用Java语言完成，在完成过程中经历了许多困难，不过好在网络上有许多大佬的经验。在参考许多大佬的代码与设计思路下，完成了此聊天室，不过还不够完善。

目前本聊天系统还有待完善的功能如下: 

- 注册界面相对比较单一, 不够详细, 不能设置个性签名，个人简介等等。 

- 聊天内容不能保存, 退出聊天室之后就找不到之前的聊天内容, 如果在聊天过程中有比较重要的内容需要更慎重的记录。 

- 不能像QQ那样发送图片表情, 而且随意添加在聊天内容的任何位置。

- 不能让窗口抖动

  希望在以后的学习中，能够完善此项目。