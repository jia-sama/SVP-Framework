#!/bin/bash

echo "=================================================="
echo "🚀 SVP-Framework Auto Git Push Pipeline Started..."
echo "=================================================="

# 1. 显示当前状态
echo -e "\n[*] Current Status:"
git status -s

# 2. 将所有更改加入暂存区
echo -e "\n[*] Adding all changes to staging area..."
git add .

# 3. 提示输入 Commit 信息（如果不输入，默认使用 "auto-update"）
echo -e "\n"
read -p "📝 Enter commit message (Press Enter to use default 'auto-update'): " msg

if [ -z "$msg" ]; then
  msg="chore: auto-update codebase and figures"
fi

echo "[*] Committing with message: '$msg'..."
git commit -m "$msg"

# 4. 推送到远程 Github 仓库的 main 分支
echo -e "\n[*] Pushing to Github (origin/main)..."
git push origin main

echo -e "\n=================================================="
echo "✅ [SUCCESS] Codebase successfully pushed to Github!"
echo "=================================================="