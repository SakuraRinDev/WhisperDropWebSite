# WhisperDrop Visual Effects 技術リファレンス

このドキュメントは、WhisperDropサイトの開発過程で検証・実装した視覚効果技術をまとめた指南書です。
各セクションでは、技術の原理、使用したコード、得られた効果、および判明した制約を記載しています。

---

## 目次

1. [AI深度推定 (Depth-Anything-V2)](#1-ai深度推定-depth-anything-v2)
2. [2.5Dパララックス — Canvas2Dマルチレイヤー方式](#2-25dパララックス--canvas2dマルチレイヤー方式)
3. [2.5Dパララックス — WebGL POM方式（採用）](#3-25dパララックス--webgl-pom方式採用)
4. [深度マップ生成ユーティリティ](#4-深度マップ生成ユーティリティ)
5. [クリックリップル — WebGLオーバーレイ（採用）](#5-クリックリップル--webglオーバーレイ採用)
6. [DOMディストーション — html2canvas + WebGL](#6-domディストーション--html2canvas--webgl)
7. [DOMディストーション — SVG feDisplacementMap](#7-domディストーション--svg-fedisplacementmap)
8. [DOMディストーション — 調査結果と結論](#8-domディストーション--調査結果と結論)

---

## 1. AI深度推定 (Depth-Anything-V2)

**ファイル:** `tools/gen-depth.html`, `2.5D_Parallax_HQ.html`

### 概要
任意の2D画像から深度マップ（奥行き情報）を推定するAIモデル。
ブラウザ上で Transformers.js を使い、ONNXモデルを WebGPU/WASM で実行する。

### 使用モデル
```
Primary:   onnx-community/depth-anything-v2-small (~50MB)
Fallback:  Xenova/dpt-hybrid-midas
Device:    WebGPU → WASM (フォールバック)
```

### コア処理
```javascript
import { pipeline } from '@huggingface/transformers';

// モデルロード
const estimator = await pipeline('depth-estimation',
  'onnx-community/depth-anything-v2-small',
  { device: 'webgpu' }
);

// 推定実行
const output = await estimator(imageDataUrl);
const depthMap = output.depth;
// depthMap.data: Uint8Array (0=far/黒, 255=near/白)
// depthMap.width, depthMap.height: モデル出力解像度
```

### バイリニア補間によるアップスケーリング
モデル出力は元画像より低解像度のため、元サイズに拡大する:

```javascript
function bilinearUpscale(src, srcW, srcH, targetW, targetH) {
  const out = new Uint8Array(targetW * targetH);
  for (let y = 0; y < targetH; y++) {
    const sy = (y / targetH) * srcH;
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, srcH - 1);
    const fy = sy - y0;
    for (let x = 0; x < targetW; x++) {
      const sx = (x / targetW) * srcW;
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, srcW - 1);
      const fx = sx - x0;
      const a = src[y0 * srcW + x0], b = src[y0 * srcW + x1];
      const c = src[y1 * srcW + x0], d = src[y1 * srcW + x1];
      out[y * targetW + x] = (a * (1-fx) + b * fx) * (1-fy)
                            + (c * (1-fx) + d * fx) * fy;
    }
  }
  return out;
}
```

### 制約
- モデルダウンロード: 初回~50MB（ブラウザにキャッシュされる）
- 推論時間: 1280x720で約1〜2秒
- モバイルでは推論+レイヤー生成でメモリ不足→クラッシュの原因に

---

## 2. 2.5Dパララックス — Canvas2Dマルチレイヤー方式

**ファイル:** `2.5D_Parallax_HQ.html`

### 概要
深度マップを元に画像をN枚のレイヤーに分離し、各レイヤーを `translate3d` で
奥行きに応じた量だけ動かすことで、2.5D効果を生み出す。

### レイヤー生成 — Hannウィンドウブレンド
各レイヤーは深度範囲に基づき、滑らかなアルファブレンドで分離される:

```javascript
const N = 32; // レイヤー数
for (let i = 0; i < N; i++) {
  const center = (i + 0.5) / N;  // レイヤーの深度中心 (0=遠, 1=近)
  const lo = center - 1.0 / N;
  const hi = center + 1.0 / N;

  for (let p = 0; p < totalPixels; p++) {
    const dN = depthData[p] / 255;  // 正規化深度
    if (dN < lo || dN > hi) { alpha = 0; continue; }

    // Hannウィンドウ — レイヤー境界で滑らかに0になる
    const t = (dN - lo) / (hi - lo);
    alpha = Math.sin(t * Math.PI);
  }
}
```

### アニメーションループ
```javascript
function animate() {
  mouseX += (targetX - mouseX) * 0.09; // スムージング

  for (let i = 0; i < N; i++) {
    const d = layerDepths[i];
    const factor = Math.pow(d, 1.2);  // 非線形パララックス（近い物体ほど大きく動く）
    const tx = offsetX * strength * factor;
    const ty = offsetY * strength * factor * 0.75; // 縦方向は控えめ
    const tz = (d - 0.5) * 80;        // Z軸分離（CSS 3D空間）
    layers[i].style.transform = `translate3d(${tx}px, ${ty}px, ${tz}px)`;
  }

  // シーン全体の回転
  const rotY = offsetX * rotation;
  const rotX = -offsetY * rotation * 0.65;
  container.style.transform = `rotateY(${rotY}deg) rotateX(${rotX}deg)`;

  requestAnimationFrame(animate);
}
```

### 設定値
```javascript
depthStrength: 500   // パララックス変位量（px）
rotation:      9     // 回転角度（deg）
layerCount:    32    // レイヤー枚数
scale:         1.07  // カバーフィット拡大率
```

### 制約と教訓
- **メモリ消費**: 32レイヤー × 1280×720 = ~118MB → モバイルでクラッシュ
- レイヤー数を減らすと段差（バンディング）が目立つ
- Canvas2Dの多重合成は CPU 負荷が高い
- **結論**: WebGL POM方式に移行 → 単一GPUパスで同等以上の品質

---

## 3. 2.5Dパララックス — WebGL POM方式（採用）

**ファイル:** `tools/2.5D_Parallax_WebGL.html`, `index.html`

### 概要
Parallax Occlusion Mapping (POM) — 深度マップに沿ってレイマーチングし、
単一のGPUパスで連続的なパララックスを実現する。Canvas2D方式の完全な代替。

### フラグメントシェーダー（核心部分）
```glsl
precision mediump float;
varying vec2 vUV;
uniform sampler2D uImage;       // hero.png
uniform sampler2D uDepth;       // hero-depth.png
uniform vec2  uMouse;           // -0.5..0.5
uniform vec2  uResolution;      // キャンバスサイズ
uniform float uStrength;        // 160 (変位ピクセル数)
uniform float uSteps;           // 34 (レイマーチステップ数)

void main() {
  vec2 uv = vUV;
  vec2 pxToUV = vec2(1.0) / uResolution;
  vec2 maxOffset = -uMouse * uStrength * pxToUV;

  // POM レイマーチ
  float layerStep = 1.0 / uSteps;
  vec2 deltaUV = maxOffset * layerStep;

  vec2 curUV = uv;
  float curLayerDepth = 0.0;
  float sampledHeight = 1.0 - texture2D(uDepth, curUV).r;
  // 深度反転: Depth-Anything は 1=近, POM は 1=遠

  for (int i = 0; i < 64; i++) {
    if (float(i) >= uSteps) break;   // uniform で制御
    if (curLayerDepth >= sampledHeight) break;  // 表面に到達
    curUV += deltaUV;
    sampledHeight = 1.0 - texture2D(uDepth, curUV).r;
    curLayerDepth += layerStep;
  }

  // 線形補間リファインメント（階段アーティファクト防止）
  vec2 prevUV = curUV - deltaUV;
  float afterDepth = sampledHeight - curLayerDepth;
  float beforeDepth = (1.0 - texture2D(uDepth, prevUV).r)
                    - (curLayerDepth - layerStep);
  float weight = afterDepth / (afterDepth - beforeDepth);
  vec2 finalUV = mix(curUV, prevUV, weight);

  finalUV = clamp(finalUV, vec2(0.0), vec2(1.0));
  gl_FragColor = texture2D(uImage, finalUV);
}
```

### POMアルゴリズム解説
1. マウス位置からパララックスのオフセット方向を計算
2. UV空間をステップ数（34段）に分割して、深度フィールドに沿ってレイマーチ
3. 各ステップでレイの高さ (`curLayerDepth`) を深度マップの値と比較
4. レイが「表面」にぶつかったら停止
5. 最後の2サンプル間で線形補間 → 滑らかな結果

### Canvas2D方式との比較
| 項目 | Canvas2D | WebGL POM |
|------|----------|-----------|
| メモリ | ~118MB (32レイヤー) | ~6MB (2テクスチャ) |
| 品質 | レイヤー境界で段差あり | 連続的、段差なし |
| GPU負荷 | CPU描画 | 軽量（単一パス） |
| モバイル | クラッシュ | 安定動作 |

### 採用時の設定値
```javascript
// デスクトップ・モバイル共通
const SETTINGS = {
  depthStrength: 160,
  rotation: 4,
  scale: 1,
  steps: 34
};
```

### 入力ハンドリング
```javascript
// マウス → -0.5..0.5 に正規化
document.addEventListener('mousemove', e => {
  targetX = (e.clientX / innerWidth) - 0.5;
  targetY = (e.clientY / innerHeight) - 0.5;
});

// ジャイロ (モバイル) → iOS 13+ パーミッション対応
window.addEventListener('deviceorientation', e => {
  targetX = Math.max(-1, Math.min(1, (e.gamma || 0) / 35)) * 0.5;
  targetY = Math.max(-1, Math.min(1, ((e.beta || 0) - 40) / 35)) * 0.5;
});

// スムージング (レンダーループ内)
smoothX += (targetX - smoothX) * 0.08;
smoothY += (targetY - smoothY) * 0.08;
```

---

## 4. 深度マップ生成ユーティリティ

**ファイル:** `tools/gen-depth.html`

### 概要
`hero.png` を読み込み、Depth-Anything-V2で深度推定し、
`hero-depth.png` としてダウンロードする単体ツール。

### 使い方
1. ブラウザで `tools/gen-depth.html` を開く（HTTPSまたはlocalhostが必要）
2. 自動的に `../hero.png` を読み込み、深度推定を実行
3. 完了すると `hero-depth.png` が自動ダウンロードされる

### 出力仕様
- フォーマット: PNG (グレースケール)
- サイズ: hero.png と同じ解像度 (1344x896)
- 深度規約: 0=遠い（黒）, 255=近い（白）
- ファイルサイズ: 約297KB

### 用途
hero.png を差し替えた際に、対応する深度マップを再生成する。
ランタイムでのAI推論を不要にし、パフォーマンスとモバイル安定性を確保。

---

## 5. クリックリップル — WebGLオーバーレイ（採用）

**ファイル:** `tools/ripple-test.html`, `index.html`

### 概要
画面全体を覆う透明なWebGLキャンバス上に、クリック位置から
同心円状に広がる白い波紋を描画する。`pointer-events: none` により
下のDOM操作を妨げない。

### フラグメントシェーダー
```glsl
precision mediump float;
uniform vec2  uRes;
uniform float uTime;
uniform vec3  uRip[8];   // xy=中心位置, z=発生時刻
uniform float uSpd, uFreq, uAmp, uDec, uThk, uOpa;

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float asp = uRes.x / uRes.y;
  float a = 0.0;

  for (int i = 0; i < 8; i++) {
    if (uRip[i].z < 0.0) continue;           // 未使用スロット
    float age = uTime - uRip[i].z;
    if (age < 0.0 || age > 4.0) continue;     // 寿命切れ

    vec2 d = uv - uRip[i].xy;
    d.x *= asp;                                // アスペクト比補正
    float dist = length(d);

    // 拡大するリング
    float rad = age * uSpd * 0.35;
    float rd = abs(dist - rad);

    // ガウシアンリング（波面のシャープネス）
    float ring = exp(-rd * rd / (uThk * uThk * 0.01));

    // 同心サブリング
    float w = 0.5 + 0.5 * cos(dist * uFreq - age * uSpd * 8.0);

    // 合成 + 時間減衰
    a += ring * w * uAmp * exp(-age * uDec);
  }

  a = clamp(a, 0.0, 1.0) * uOpa;
  gl_FragColor = vec4(1.0, 1.0, 1.0, a);  // 白 + 可変アルファ
}
```

### 波紋の数学
- **波面**: `radius = age × speed × 0.35` → 時間とともに拡大
- **リングシャープネス**: ガウス曲線 `exp(-距離²/厚み²)`
- **サブリング**: コサイン波 `cos(距離 × 周波数 - 時間 × 速度)`
- **減衰**: 指数関数 `exp(-age × decay)`
- **合成**: 加算ブレンド (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`)
- **同時リップル**: 最大8個（リングバッファ）

### 採用時の設定値
```javascript
const RIPPLE = {
  speed:     0.6,
  frequency: 35,
  amplitude: 2.1,
  decay:     0.9,
  thickness: 1.4,
  opacity:   0.35
};
```

### HTMLへの組み込み方
```html
<!-- bodyの末尾に追加 -->
<canvas id="ripple-canvas"
  style="position:fixed;top:0;left:0;width:100vw;height:100vh;
         pointer-events:none;z-index:99999;"></canvas>
```

---

## 6. DOMディストーション — html2canvas + WebGL

**ファイル:** `tools/distortion-test.html`

### 概要
ページ全体のスクリーンショットを html2canvas で撮影し、それをWebGLテクスチャとして
読み込んでディストーションシェーダーで歪ませる。テキスト・画像・カードすべてが
ピクセル単位で歪む。

### フラグメントシェーダー
```glsl
precision mediump float;
varying vec2 vUV;
uniform sampler2D uTex;       // ページのスクリーンショット
uniform vec2  uClick;          // クリック位置 (0..1)
uniform float uTime;           // 経過時間
uniform float uFreq, uAmp, uSpeed, uDecay, uAspect;

void main() {
  vec2 uv = vUV;
  vec2 diff = uv - uClick;
  diff.x *= uAspect;            // アスペクト比補正
  float dist = length(diff);

  if (dist > 0.001 && uTime > 0.0) {
    // 拡大する波面（ガウス集中）
    float wavefront = uTime * uSpeed * 0.08;
    float nearFront = exp(-(dist - wavefront)² * 40.0);

    // サイン波振動
    float wave = sin(dist * uFreq - uTime * uSpeed);

    // 時間減衰
    float fade = exp(-uTime * uDecay);

    // 放射方向に変位
    vec2 dir = normalize(diff);
    uv += dir * wave * nearFront * fade * uAmp;
  }

  uv = clamp(uv, 0.0, 1.0);
  gl_FragColor = texture2D(uTex, uv);
}
```

### 処理フロー
```
クリック → html2canvas(document.body) → Canvas要素
         → gl.texImage2D() でテクスチャ化
         → WebGLキャンバスを表示 (pointer-events: none)
         → 2秒間ディストーションアニメーション
         → WebGLキャンバスを非表示
```

### 設定値
```javascript
const CONFIG = {
  frequency: 35,       // 波の細かさ
  amplitude: 0.015,    // 歪み強度（テクセル単位）
  speed:     5.0,      // 波の伝搬速度
  decay:     1.8,      // 減衰指数
  duration:  2.0       // 持続時間（秒）
};
```

### 注意: Y軸反転
html2canvas は左上原点、WebGL は左下原点のため、頂点シェーダーで反転が必要:
```glsl
uv.y = 1.0 - uv.y;  // html2canvas → GL座標変換
```

### 判明した問題点
1. **レイアウトシフト**: html2canvas がDOMをクローンする際にスクロールバーが変動
2. **スクロール停止**: キャプチャ中に `overflow` が操作される
3. **パララックスリセット**: WebGLコンテキストが影響を受け、ヒーロー画像の角度が0に戻る
4. **パフォーマンス**: キャプチャに50-200ms、その間UIがフリーズ
5. **スクロール位置ズレ**: 事前キャプチャだとスクロール後に古い画像が表示される

### 結論
**採用見送り。** 効果としては理想的（ピクセル単位のラジアルリップル）だが、
html2canvas の副作用が実用レベルで解決不可能。

---

## 7. DOMディストーション — SVG feDisplacementMap

**ファイル:** `tools/distortion-svg-test.html`

### 概要
SVGフィルター `feDisplacementMap` をDOMラッパーに直接適用し、
キャプチャなしでテキスト・画像をピクセル単位で歪ませる。

### SVGフィルター定義
```xml
<svg style="position:absolute;width:0;height:0;">
  <filter id="ripple-distort" x="-10%" y="-10%" width="120%" height="120%">
    <feTurbulence id="turb" type="turbulence"
      baseFrequency="0.02" numOctaves="3" seed="1" result="noise"/>
    <feDisplacementMap id="displace" in="SourceGraphic" in2="noise"
      scale="0" xChannelSelector="R" yChannelSelector="G"/>
  </filter>
</svg>
```

### 適用方法
```javascript
wrapper.style.filter = 'url(#ripple-distort)';
```

### アニメーション
```javascript
document.addEventListener('click', (e) => {
  // ノイズパターンをランダム化
  turb.setAttribute('seed', Math.floor(Math.random() * 999));
  wrapper.style.filter = 'url(#ripple-distort)';
  const start = performance.now();

  function tick() {
    const t = (performance.now() - start) / duration;
    if (t >= 1) {
      displace.setAttribute('scale', '0');
      wrapper.style.filter = 'none';
      return;
    }

    // クイックアタック + スロウディケイ
    const envelope = t < 0.15
      ? t / 0.15                            // 立ち上がり (0→1)
      : Math.pow(1 - (t - 0.15) / 0.85, 2); // 二次減衰 (1→0)

    displace.setAttribute('scale', peakScale * envelope);
    requestAnimationFrame(tick);
  }
  tick();
});
```

### 設定値
```javascript
const CONFIG = {
  peakScale:     30,    // 最大歪み量
  duration:      600,   // ミリ秒
  baseFrequency: 0.02,  // ノイズスケール
  numOctaves:    3      // ノイズ詳細度
};
```

### 判明した問題点
1. **放射状リップル不可**: feTurbulence は均一なノイズパターンのため、
   クリック位置から広がる同心円状の歪みは原理的に実現できない
2. **画面全体が同時に揺れる**: 波面の伝搬という概念が表現できない
3. **動的 feImage の問題**: Canvas要素を `feImage href="#canvasId"` で参照する方法は
   Firefox (bug 455986, 2008年〜未修正) と Safari で動作しない

### 結論
**採用見送り。** キャプチャ不要で軽量だが、「クリック位置から放射状に広がる波紋」という
目標の効果を実現できない。画面全体が均一に揺れるだけ。

---

## 8. DOMディストーション — 調査結果と結論

### 調査した全アプローチ

| アプローチ | DOM歪み | クロスブラウザ | キャプチャ不要 | 放射状 | 実用性 |
|-----------|---------|-------------|-------------|-------|-------|
| html2canvas + WebGL | OK (ピクセル) | OK | NG | OK | 副作用多い |
| SVG feDisplacementMap (feTurbulence) | OK (ピクセル) | OK | OK | NG | 均一揺れのみ |
| SVG feDisplacementMap (Canvas参照) | OK (ピクセル) | NG | OK | OK | FF/Safari壊れてる |
| CSS transform per-element | 粗い | OK | OK | OK | 効果が違う |
| Curtains.js | 画像のみ | OK | OK | OK | テキスト対象外 |
| PixiJS DisplacementFilter | 画像のみ | OK | OK | OK | テキスト対象外 |
| CSS Houdini Paint API | NG | Chrome only | OK | - | 用途が違う |
| getDisplayMedia (Screen Capture) | OK | 一部 | NG (許可必要) | OK | UX摩擦大 |
| CSS backdrop-filter + SVG | NG | NG | OK | - | 壊れてる |

### 根本的な壁
> ブラウザには「今表示されているピクセル」を軽量にテクスチャとして
> GPUに渡すAPIが存在しない。

- WebGLはDOM要素を読めない（テクスチャ化には html2canvas 等が必要）
- html2canvasはDOMクローンの副作用が避けられない
- SVGフィルターの動的ソース参照はクロスブラウザで壊れている
- React/Next.js等のフレームワーク変更では解決しない（ブラウザAPIの制約）

### 最終判断
**リップルオーバーレイ（白い輪の重畳）のみ採用。**
ピクセル単位のDOMディストーションは現時点のWeb技術では安定して実現できない。

---

## 参考リンク

### CodePen 実例
- [Ripple (SVG filter + CSS)](https://codepen.io/yuanchuan/pen/NLRxvN)
- [Awesome SVG Ripple Effect](https://codepen.io/T-P/pen/ZWyLje)
- [SVG waves with feDisplacementMap](https://codepen.io/enxaneta/post/svg-waves-with-fedisplacementmap)
- [GSAP SVG feDisplacementMap](https://codepen.io/jonathan/pen/NqZPwd)
- [SVG Displacement Map](https://codepen.io/osublake/pen/WQyBJb)

### チュートリアル・記事
- [Codrops: feDisplacementMap でテキストを歪ませる](https://tympanus.net/codrops/2019/02/12/svg-filter-effects-conforming-text-to-surface-texture-with-fedisplacementmap/)
- [Codrops: SVGフィルターで画像ディストーション](https://tympanus.net/codrops/2019/03/12/image-distortion-effects-with-svg-filters/)
- [Codrops: ボタンのディストーション効果](https://tympanus.net/codrops/2016/05/11/distorted-button-effects-with-svg-filters/)
- [Smashing Magazine: feDisplacementMap 詳解](https://www.smashingmagazine.com/2021/09/deep-dive-wonderful-world-svg-displacement-filtering/)
- [GitHub: SVG water ripple example](https://github.com/chenxiaochun/svg-living-example/blob/master/examples/water-ripple.html)

### ライブラリ
- [Transformers.js](https://huggingface.co/docs/transformers.js/) — ブラウザ上AI推論
- [Curtains.js](https://www.curtainsjs.com/) — DOM→WebGLマッピング（画像のみ）
- [html2canvas](https://html2canvas.hertzen.com/) — DOM→Canvas変換
- [jquery.ripples (sirxemic)](https://github.com/sirxemic/jquery.ripples) — 背景画像の水面効果

---

## ファイル一覧

| ファイル | 技術 | 状態 |
|---------|------|------|
| `index.html` | WebGL POM + リップルオーバーレイ | **本番採用** |
| `hero.png` | ヒーロー画像 | 本番使用 |
| `hero-depth.png` | 事前生成された深度マップ | 本番使用 |
| `tools/gen-depth.html` | 深度マップ生成ユーティリティ | ユーティリティ |
| `tools/2.5D_Parallax_WebGL.html` | WebGL POMテストハーネス | テスト用 |
| `tools/ripple-test.html` | リップルシェーダーテスト | テスト用 |
| `tools/distortion-test.html` | html2canvas + WebGL歪み | 検証済・不採用 |
| `tools/distortion-svg-test.html` | SVG feDisplacementMap歪み | 検証済・不採用 |
| `2.5D_Parallax_HQ.html` | Canvas2D多層パララックス(旧版) | 参考保存 |
