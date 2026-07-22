# Deploy len Hugging Face Spaces

## 1. Tao Space

1. Dang nhap https://huggingface.co/new-space.
2. Dat ten Space, vi du `parking-license-ai`.
3. Chon **Docker** lam Space SDK va chon **Blank**.
4. Chon **Public** de Backend co the goi API ma khong can Hugging Face token.

## 2. Day source len Space

Tai thu muc project, thay `YOUR_USERNAME` bang tai khoan Hugging Face:

```powershell
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/parking-license-ai
git push space main
```

Neu nhanh hien tai khong phai `main`, dung:

```powershell
git push space HEAD:main
```

Hugging Face co the yeu cau Access Token thay cho mat khau. Tao token co quyen write tai https://huggingface.co/settings/tokens.

## 3. Kiem tra

Cho den khi Space hien thi **Running**, sau do mo:

```text
https://YOUR_USERNAME-parking-license-ai.hf.space/health
https://YOUR_USERNAME-parking-license-ai.hf.space/docs
```

Backend goi:

```http
POST https://YOUR_USERNAME-parking-license-ai.hf.space/predict-file-fast
Content-Type: multipart/form-data
```

Field file anh phai co ten `file`. Vi du:

```bash
curl -X POST "https://YOUR_USERNAME-parking-license-ai.hf.space/predict-file-fast" \
  -F "file=@vehicle.jpg"
```

## Luu y

- File `models/best.pt` phai duoc commit cung source.
- Lan build dau va lan goi dau co the cham vi PyTorch, YOLO va EasyOCR can duoc tai vao bo nho.
- Goi mien phi co the sleep khi khong su dung; hay goi `/health` va `/predict-file-fast` truoc buoi demo.
