# Hexo Theme Solo

一个简洁、优雅的Hexo博客主题，支持暗色模式切换。

## 特性

- 响应式设计，适配各种设备
- 暗色/亮色模式切换
- 简洁的文章列表展示
- 分类、标签、归档页面
- 个人资料展示
- 支持评论系统

## 安装

```bash
cd your-hexo-site
git clone https://github.com/your-username/hexo-theme-solo.git themes/solo
```

## 配置

修改Hexo根目录下的 `_config.yml` 文件：

```yaml
theme: solo
```

## 主题配置

编辑 `themes/solo/_config.yml` 文件，根据需要修改配置项：

```yaml
# 菜单名称
menuname: Solo Blog

# 作者昵称
author: Your Name

# 主菜单导航
menu:
  GitHub: https://github.com/your-username
  笔记源: https://github.com/your-username/notes

# 顶部菜单
topmenu:
  首页: /
  笔记: /notes
  分类: /categories
  标签: /tags
  归档: /archives
  关于: /about

# 网站描述
description: 记录学习与生活的点滴

# 网站关键词
keywords: blog,hexo,solo,coding

# 博客头像
img_src: your-avatar-url

# 标语
words: 平静前行，不负时光
```

## 许可证

[MIT](LICENSE)