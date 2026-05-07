# gh CLI + GitHub Release 视频托管流程

## 为什么用 Release assets

- GitHub README 可以 `<img src="xxx.mp4">` 但**多数浏览器不会原生播放**（只是下载链接）。
- 上传到 GitHub Release 后，文件会有一个 CDN URL：
  `https://github.com/<user>/<repo>/releases/download/<tag>/<file>`
  这个 URL 在 README 里 `<video>` 标签**能嵌入原生播放器**（或者 `<img>` 配合 GIF 缩略图 → 点击 MP4 URL 用浏览器内置播放器）。
- 而且 **Release assets 不占 repo 大小限制**（单文件 ≤2GB）。

## 本地 gh CLI 已装

```bash
/home/ubuntu/mambaforge/bin/gh
gh version 2.92.0
```

通过 `conda install -n base -c conda-forge gh` 安装（SSH 到 github.com 可通，但 HTTPS 被防火墙/DNS 卡住，所以直接下载二进制不行）。

## 认证：待用户提供 PAT

我没有 GitHub Personal Access Token（PAT）。请用户完成这一步：

1. 浏览器打开 https://github.com/settings/tokens
2. "Generate new token" → **classic**（简单）
3. 权限勾选：**`repo`**（全部子项） + **`write:packages`**（如果想 push 镜像）
4. 生成后复制 token（格式：`ghp_xxxxxxxxxxxxxxxxxxxx`）
5. 在本机终端执行：

```bash
echo 'ghp_xxxxxxxxxxxxxxxxxxxx' | gh auth login --with-token
gh auth status  # 应显示 ✓ Logged in to github.com as lz-googlefycy
```

或者用环境变量：

```bash
export GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
gh auth status  # 应显示 ✓ Logged in to github.com as lz-googlefycy (GH_TOKEN)
```

## 认证完成后：一键上传 4-suite MP4 到 Release

```bash
cd /home/ubuntu/openvla-libero

# Create release tag + upload 5 MB 4-suite demo
gh release create v0.1-demos \
    --title "Demo videos v0.1 — OpenVLA on LIBERO 4 suites" \
    --notes "Pre-cut highlight videos from the 400-rollout reproduction run.
- openvla_libero_4suite_demo.mp4: 40 clips across all 4 suites (5 MB)
- openvla_libero_spatial_demo.mp4: 10 clips, Spatial only (1 MB)" \
    --repo lz-googlefycy/openvla-libero \
    assets/demos/openvla_libero_4suite_demo.mp4 \
    assets/demos/openvla_libero_spatial_demo.mp4

# Same for vla-lab
cd /home/ubuntu/vla-lab
gh release create v0.1-demos \
    --title "Demo videos v0.1" \
    --notes "LIBERO reproduction demos + preview GIFs." \
    --repo lz-googlefycy/vla-lab \
    assets/demos/openvla_libero_4suite_demo.mp4 \
    assets/demos/openvla_libero_spatial_demo.mp4 \
    assets/demos/openvla_libero_3suite_demo.mp4
```

上传成功后，URL 会是类似：

```
https://github.com/lz-googlefycy/openvla-libero/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4
```

## README 更新：用 `<video>` 标签嵌入

```html
<!-- 替换现有 <a href="assets/..."><img src=".gif"></a> 模式 -->
<video controls width="480" poster="assets/demos/hero_static.jpg">
  <source src="https://github.com/lz-googlefycy/openvla-libero/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
  <a href="https://github.com/lz-googlefycy/openvla-libero/releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4">Download the 4-suite demo MP4 (5 MB)</a>
</video>
```

**注意**：GitHub README 的 HTML 实际上会 strip 掉 `<video>` 标签（出于 XSS 原因），所以**`<video>` 不会生效**。最优方案是：

- **保留当前 GIF 方案**（已经很好）
- **在文末加一个 "Direct MP4 download" section**，列出所有 Release URL

```markdown
### Direct MP4 downloads (no GIF loading)

- [4-suite demo (5 MB, 40 clips, 4 min)](https://github.com/.../releases/download/v0.1-demos/openvla_libero_4suite_demo.mp4)
- [Spatial only (1 MB, 10 clips, 1 min)](https://github.com/.../releases/download/v0.1-demos/openvla_libero_spatial_demo.mp4)
```

## Next action

用户把 PAT 发到聊天里（或我帮他生成一个命令行模板），我就能完成 release + README 更新。
