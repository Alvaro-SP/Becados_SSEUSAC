@echo off
echo %~dp0

git add .
git commit -m "connect to google sheets data"
git push
