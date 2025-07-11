<template>
  <div class="sentiment-analysis-container">
    <div class="header">
      <h1>中文情感分析系统</h1>
      <div class="dataset-selector">
        <label for="dataset">选择数据集：</label>
        <select id="dataset" v-model="selectedDataset" @change="changeDataset">
          <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">
            {{ dataset.name }}
          </option>
        </select>
      </div>
    </div>

    <div class="main-content">
      <!-- 左侧数据分布区域 -->
      <div class="left-section">
        <div class="chart-container">
          <h2>{{ selectedDatasetName }}数据集分布</h2>
          <div ref="distributionChart" class="chart"></div>
        </div>
        <div class="stats-container">
          <h3>数据集统计信息</h3>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="stat-label">总样本数</span>
              <span class="stat-value">{{ datasetStats.total }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">积极样本</span>
              <span class="stat-value">{{ datasetStats.positive }}</span>
              <span class="stat-percent">({{ datasetStats.positivePercent }}%)</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">消极样本</span>
              <span class="stat-value">{{ datasetStats.negative }}</span>
              <span class="stat-percent">({{ datasetStats.negativePercent }}%)</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧分析区域 -->
      <div class="right-section">
        <div class="analysis-box">
          <h2>文本情感分析</h2>
          <textarea
              v-model="inputText"
              placeholder="请输入中文文本进行情感分析..."
              rows="5"
          ></textarea>
          <div class="action-buttons">
            <button class="analyze-btn" @click="analyzeSentiment">
              <i class="icon-analysis"></i> 开始分析
            </button>
            <button class="clear-btn" @click="clearInput">
              <i class="icon-clear"></i> 清空
            </button>
          </div>

          <div v-if="analysisResult" class="result-container">
            <div class="result-header">
              <h3>分析结果</h3>
              <span class="model-info">使用模型: {{ currentModelName }}</span>
            </div>
            <div class="result-content" :class="analysisResult.sentiment === '积极' ? 'positive' : 'negative'">
              <div class="sentiment-display">
                <span class="sentiment-icon">
                  <i v-if="analysisResult.sentiment === '积极'" class="icon-happy"></i>
                  <i v-else class="icon-sad"></i>
                </span>
                <span class="sentiment-text">
                  {{ analysisResult.sentiment }}
                </span>
              </div>
              <div class="confidence-meter">
                <div class="meter-label">置信度</div>
                <div class="meter-bar">
                  <div
                      class="meter-fill"
                      :style="{ width: analysisResult.confidence * 100 + '%' }"
                  ></div>
                  <span class="meter-value">
                    {{ (analysisResult.confidence * 100).toFixed(1) }}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import axios from 'axios'
import { ElMessage, ElLoading } from 'element-plus'
export default {
  name: 'SentimentAnalysis',
  data() {
    return {
      selectedDataset: 'chn',
      datasets: [
        { id: 'chn', name: 'ChnSentiCorp', model: 'BERT-BiLSTM-Attention' },
        { id: 'waimai', name: 'Waimai_10k', model: 'BERT-BiLSTM-Attention' },
      ],
      datasetDistribution: {
        chn: { total: 7766, positive: 5322, negative: 2444 },
        waimai: { total: 11986, positive: 7986, negative: 4000 },
      },
      inputText: '',
      analysisResult: null,
      distributionChart: null,
      datasetStats: {
        total: 0,
        positive: 0,
        negative: 0,
        positivePercent: 0,
        negativePercent: 0
      }
    }
  },
  computed: {
    selectedDatasetName() {
      const dataset = this.datasets.find(d => d.id === this.selectedDataset)
      return dataset ? dataset.name : ''
    },
    currentModelName() {
      const dataset = this.datasets.find(d => d.id === this.selectedDataset)
      return dataset ? dataset.model : ''
    }
  },
  mounted() {
    this.initChart()
    this.updateDatasetStats()
  },
  methods: {
    initChart() {
      this.distributionChart = echarts.init(this.$refs.distributionChart)
      this.updateDistributionChart()
      window.addEventListener('resize', this.handleResize)
    },
    handleResize() {
      this.distributionChart.resize()
    },
    updateDistributionChart() {
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
          orient: 'vertical',
          left: 10,
          data: ['积极', '消极']
        },
        series: [
          {
            name: '情感分布',
            type: 'pie',
            radius: ['50%', '70%'],
            avoidLabelOverlap: false,
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '18',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: [
              {
                value: this.datasetStats.positive,
                name: '积极',
                itemStyle: { color: '#4CAF50' }
              },
              {
                value: this.datasetStats.negative,
                name: '消极',
                itemStyle: { color: '#F44336' }
              }
            ]
          }
        ]
      }
      this.distributionChart.setOption(option)
    },
    updateDatasetStats() {
      const data = this.datasetDistribution[this.selectedDataset]
      this.datasetStats = {
        total: data.total,
        positive: data.positive,
        negative: data.negative,
        positivePercent: ((data.positive / data.total) * 100).toFixed(1),
        negativePercent: ((data.negative / data.total) * 100).toFixed(1)
      }
      this.updateDistributionChart()
    },
    async analyzeSentiment() {
      if (!this.inputText.trim()) {
        this.$message.error('请输入要分析的文本')
        return
      }

      try {
        const response = await axios.post('http://localhost:5000/api/analyze', {
          text: this.inputText,
          dataset: this.selectedDataset  // 传入当前选择的数据集类型
        })
        this.analysisResult = response.data
        this.$message.success('分析完成')
      } catch (error) {
        console.error('情感分析失败:', error)
        this.$message.error('分析失败，请重试')
      }
    },
    clearInput() {
      this.inputText = ''
      this.analysisResult = null
    },
    changeDataset() {
      this.updateDatasetStats()
      this.analysisResult = null
    },
    beforeDestroy() {
      window.removeEventListener('resize', this.handleResize)
    }
  }
}
</script>


<style scoped>
.sentiment-analysis-container {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.header {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e8e8e8;
}

.header h1 {
  font-size: 24px;
  color: #333;
  margin: 0;
}

.main-content {
  display: flex;
  gap: 24px;
}

.left-section {
  flex: 1;
  min-width: 400px;
}

.right-section {
  flex: 1;
  min-width: 400px;
}

.chart-container {
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  margin-bottom: 16px;
}

.chart-container h2 {
  font-size: 16px;
  color: #333;
  margin: 0 0 16px 0;
  font-weight: 500;
}

.chart {
  width: 100%;
  height: 300px;
}

.stats-container {
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.stats-container h3 {
  font-size: 16px;
  color: #333;
  margin: 0 0 16px 0;
  font-weight: 500;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px;
  background: #fafafa;
  border-radius: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 20px;
  font-weight: 500;
  color: #333;
}

.stat-percent {
  font-size: 12px;
  color: #888;
  margin-top: 4px;
}

.analysis-box {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.analysis-box h2 {
  font-size: 16px;
  color: #333;
  margin: 0 0 16px 0;
  font-weight: 500;
}

.analysis-box textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  resize: none;
  font-size: 14px;
  line-height: 1.5;
  margin-bottom: 16px;
  transition: all 0.3s;
}

.analysis-box textarea:focus {
  outline: none;
  border-color: #40a9ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.action-buttons {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.analyze-btn {
  padding: 10px 16px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 6px;
}

.analyze-btn:hover {
  background-color: #40a9ff;
}

.clear-btn {
  padding: 10px 16px;
  background-color: #f5f5f5;
  color: #666;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 6px;
}

.clear-btn:hover {
  background-color: #e8e8e8;
}

.result-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.result-header {
  margin-bottom: 16px;
}

.result-header h3 {
  font-size: 16px;
  color: #333;
  margin: 0;
  font-weight: 500;
}

.result-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 24px;
  border-radius: 8px;
}

.result-content.positive {
  background-color: rgba(76, 175, 80, 0.08);
}

.result-content.negative {
  background-color: rgba(244, 67, 54, 0.08);
}

.sentiment-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 24px;
}

.sentiment-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.sentiment-icon .icon-happy {
  color: #4CAF50;
}

.sentiment-icon .icon-sad {
  color: #F44336;
}

.sentiment-text {
  font-size: 24px;
  font-weight: 500;
}

.positive .sentiment-text {
  color: #4CAF50;
}

.negative .sentiment-text {
  color: #F44336;
}

.confidence-meter {
  width: 100%;
  max-width: 300px;
}

.meter-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
  text-align: center;
}

.meter-bar {
  height: 24px;
  background-color: #f5f5f5;
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}

.meter-fill {
  height: 100%;
  transition: width 0.6s ease;
}

.positive .meter-fill {
  background-color: #4CAF50;
}

.negative .meter-fill {
  background-color: #F44336;
}

.meter-value {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white;
  font-size: 12px;
  font-weight: bold;
}

@media (max-width: 992px) {
  .main-content {
    flex-direction: column;
  }

  .left-section, .right-section {
    min-width: auto;
  }
}

@media (max-width: 576px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
