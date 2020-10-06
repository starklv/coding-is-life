# 网页爬虫demo

注意点:
 * 1.因为运用selenium.webdriver来打开网页,所以在使用driver.chrome()之前先下载google chrome驱动器,firfox也有相对应的教程,chrome驱动安装过程可参考网址:https://blog.csdn.net/weixin_44318830/article/details/103339273, 驱动下载地址可参考:http://npm.taobao.org/mirrors/chromedriver/85.0.4183.87/, 这个网址下载很快
 * 2.因为不同网页源代码的编写方式不同,所以解析网页内容时,抓取各个元素的方式也会不同,网易云还特别注意switch_to.frame()切换到iframe框架下才能抓取成功;另外网易云呈现搜索许嵩歌词的界面跟我几个月前又不一样了,只列出20首歌曲,所以解析网页格式时还需要修改一下,你那么聪明肯定会的啦 
 * 3.有时候我们还得获取网页账户中的cookies以寻求免验证登陆,因为有些网址不登陆只能浏览一定数量的页数,这在weiyi_spider中有体现
 * 4.我还上传了一些网页爬虫相关的基础知识文件,包括requests、lxml+xpath、selenium....有需要的朋友可以看一下
 * 5.上传了python版本和jupyter notebook版本,python源文件更小,更方便打开,notebook版本更适合当教程,但相应的文件更大,加载时间很慢
 
## weiyi_spider主要爬取的是微医网中上海三甲医院的各个科室

## lyrics_spider主要爬取的是网易云上许嵩歌曲的歌词
