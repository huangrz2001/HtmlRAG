import os



html_path = '/home/algo/hrz/db_construct/测试知识库/_全球购_发布违禁商品_信息_细则.html'
if not html_path or not os.path.exists(html_path):
    print({"result": "fail", "error": "HTML 文件下载失败"}) 
