// ========================================
// CIFAR-10 圖片分類應用程式
// ========================================

let model;
let isModelLoaded = false;

// CIFAR-10 類別（中英文對照）
const categories = [
    { en: 'airplane', zh: '飛機', icon: '✈️' },
    { en: 'automobile', zh: '汽車', icon: '🚗' },
    { en: 'bird', zh: '鳥', icon: '🐦' },
    { en: 'cat', zh: '貓', icon: '🐱' },
    { en: 'deer', zh: '鹿', icon: '🦌' },
    { en: 'dog', zh: '狗', icon: '🐕' },
    { en: 'frog', zh: '青蛙', icon: '🐸' },
    { en: 'horse', zh: '馬', icon: '🐴' },
    { en: 'ship', zh: '船', icon: '🚢' },
    { en: 'truck', zh: '卡車', icon: '🚚' }
];

// ========================================
// 載入模型
// ========================================

async function loadModel() {
    try {
        updateStatus('正在載入 AI 模型...', 'loading');
        console.log('開始載入模型...');
        
        model = await tf.loadLayersModel('model/model.json');
        
        // 暖機
        const warmup = tf.zeros([1, 32, 32, 3]);
        model.predict(warmup).dispose();
        warmup.dispose();
        
        isModelLoaded = true;
        updateStatus('✓ AI 已就緒！上傳照片開始辨識', 'ready');
        console.log('✓ 模型載入成功');
        
    } catch (error) {
        console.error('模型載入失敗:', error);
        updateStatus('✗ 模型載入失敗', 'error');
    }
}

// ========================================
// 檔案處理
// ========================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImage(file);
    }
}

function processImage(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const img = document.getElementById('preview');
        img.src = e.target.result;
        img.style.display = 'block';
        
        // 等圖片載入完成後進行預測
        img.onload = function() {
            if (isModelLoaded) {
                classifyImage(img);
            }
        };
    };
    
    reader.readAsDataURL(file);
}

// ========================================
// 拖曳上傳
// ========================================

const uploadArea = document.getElementById('uploadArea');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    }
});

// ========================================
// 圖片預處理
// ========================================

function preprocessImage(img) {
    return tf.tidy(() => {
        // 將圖片轉為張量
        let tensor = tf.browser.fromPixels(img);
        
        // 調整大小為 32x32
        tensor = tf.image.resizeBilinear(tensor, [32, 32]);
        
        // 正規化到 0-1
        tensor = tensor.div(255.0);
        
        // 增加 batch 維度 [32, 32, 3] -> [1, 32, 32, 3]
        tensor = tensor.expandDims(0);
        
        return tensor;
    });
}

// ========================================
// 圖片分類
// ========================================

async function classifyImage(img) {
    if (!isModelLoaded) {
        showAlert('模型尚未載入完成');
        return;
    }
    
    try {
        console.log('開始分類...');
        const startTime = performance.now();
        
        // 預處理圖片
        const tensor = preprocessImage(img);
        console.log('輸入張量 shape:', tensor.shape);
        
        // 進行預測
        const predictions = model.predict(tensor);
        const probabilities = await predictions.data();
        
        // 找出最高機率
        let maxProb = -1;
        let maxIndex = 0;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        const endTime = performance.now();
        console.log(`分類結果: ${categories[maxIndex].zh} (${(maxProb * 100).toFixed(1)}%)`);
        console.log(`耗時: ${(endTime - startTime).toFixed(2)}ms`);
        
        // 顯示結果
        displayResult(maxIndex, probabilities);
        
        // 清理記憶體
        tensor.dispose();
        predictions.dispose();
        
    } catch (error) {
        console.error('分類失敗:', error);
        showAlert('分類失敗！');
    }
}

// ========================================
// 顯示結果
// ========================================

function displayResult(predictedIndex, probabilities) {
    const predicted = categories[predictedIndex];
    
    // 建立機率排序
    const probArray = Array.from(probabilities).map((prob, idx) => ({
        index: idx,
        probability: prob
    }));
    probArray.sort((a, b) => b.probability - a.probability);
    
    // 顯示主要結果
    let html = `
        <div class="result-title">AI 辨識結果</div>
        
        <div class="result-main">
            <div class="result-icon">${predicted.icon}</div>
            <div class="result-label">${predicted.zh}</div>
            <div class="result-confidence">
                信心度: ${(probabilities[predictedIndex] * 100).toFixed(1)}%
            </div>
        </div>
        
        <div class="probabilities">
    `;
    
    // 顯示前 6 名
    for (let i = 0; i < Math.min(6, probArray.length); i++) {
        const item = probArray[i];
        const cat = categories[item.index];
        const percent = (item.probability * 100).toFixed(1);
        const isTop = i === 0;
        
        html += `
            <div class="prob-item ${isTop ? 'top' : ''}">
                <div class="prob-icon">${cat.icon}</div>
                <div class="prob-info">
                    <div class="prob-label">${cat.zh}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width: ${percent}%"></div>
                    </div>
                    <div class="prob-percent">${percent}%</div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    
    document.getElementById('result').innerHTML = html;
}

// ========================================
// 工具函數
// ========================================

function updateStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

function showAlert(message) {
    const alertDiv = document.createElement('div');
    alertDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0,0,0,0.85);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        font-size: 16px;
        z-index: 10000;
        max-width: 80%;
        text-align: center;
    `;
    alertDiv.textContent = message;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 2000);
}

// ========================================
// 初始化
// ========================================

console.log('=============================================');
console.log('CIFAR-10 圖片分類應用程式');
console.log('可辨識 10 種物品');
console.log('=============================================');

// 頁面載入後自動載入模型
window.addEventListener('load', () => {
    loadModel();
});
