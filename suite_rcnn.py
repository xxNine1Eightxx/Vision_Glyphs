@@
-    x = torch.from_numpy(im).permute(2,0,1).float()/255.0
+    x = torch.from_numpy(im).permute(2,0,1).to(torch.float32)/255.0
@@
-@torch.no_grad()
-def draw_boxes(im_rgb: np.ndarray, boxes: np.ndarray, labels: List[str], scores: np.ndarray, thr=0.5) -> np.ndarray:
+@torch.no_grad()
+def draw_boxes(im_rgb: np.ndarray, boxes: np.ndarray, labels: List[str], scores: np.ndarray, thr=0.5) -> np.ndarray:
+    # ensure plain float32 arrays for portability
+    boxes = np.asarray(boxes, dtype=np.float32)
+    scores = np.asarray(scores, dtype=np.float32)
@@
-        with torch.autocast("cuda" if dev.type=="cuda" else "cpu", enabled=self.cfg.amp):
+        # Disable AMP on CPU to avoid bfloat16→numpy issues
+        amp_ok = (dev.type == "cuda") and self.cfg.amp
+        with torch.autocast("cuda", enabled=amp_ok):
             fmap = self.backbone(x)  # [1,C,Hf,Wf]
             rois = roi_align_on_feature(fmap, props_t, (H, W), out_size=7)
             logits, deltas_all = self.head(rois)
             probs = F.softmax(logits, dim=1)
             conf, lab = probs.max(1)
@@
-            B, S, L = nms_per_class(pred_boxes, conf, lab, self.cfg.nms_iou, self.cfg.score_thr, self.cfg.topk)
-        names = [str(int(c.item())) if not class_names else class_names[int(c)-1] for c in L]
-        return {"boxes": B.cpu().numpy().tolist(), "scores": S.cpu().numpy().tolist(),
-                "labels": L.cpu().numpy().tolist(), "names": names}
+            # ensure fp32 tensors before numpy
+            B, S, L = nms_per_class(
+                pred_boxes.to(torch.float32),
+                conf.to(torch.float32),
+                lab,
+                self.cfg.nms_iou, self.cfg.score_thr, self.cfg.topk
+            )
+        names = [str(int(c.item())) if not class_names else class_names[int(c)-1] for c in L]
+        return {
+            "boxes": B.detach().cpu().to(torch.float32).numpy().tolist(),
+            "scores": S.detach().cpu().to(torch.float32).numpy().tolist(),
+            "labels": L.detach().cpu().numpy().tolist(),
+            "names": names
+        }
@@
-    if args.mode == "rcnn":
+    if args.mode == "rcnn":
         model = RCNNDetector(RCNNConfig()).to(dev).eval()
-        if not args.image:
-            print("Provide --image for inference"); return
+        if not args.image:
+            print("Provide --image for inference"); return
+        if not os.path.exists(args.image):
+            raise FileNotFoundError(f"Image path does not exist: {args.image}")
         out = model.predict_image(args.image, class_names)
@@
-    if args.out and args.image:
-        vis = draw_boxes(np.array(Image.open(args.image).convert("RGB")) if not HAS_CV2 else load_rgb(args.image)[1],
-                         np.array(out["boxes"]), out["names"], np.array(out["scores"]), thr=args.score_thr)
+    if args.out and args.image:
+        base_im = load_rgb(args.image)[1] if HAS_CV2 else np.array(Image.open(args.image).convert("RGB"))
+        vis = draw_boxes(base_im, out["boxes"], out["names"], out["scores"], thr=args.score_thr)
         if HAS_CV2:
             cv2.imwrite(args.out, vis[:, :, ::-1])
         else:
             Image.fromarray(vis).save(args.out)
         print(f"Saved: {args.out}")
