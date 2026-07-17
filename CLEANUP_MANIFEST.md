# Movie Comments 清理执行记录

生成及执行日期：2026-07-13  
审计范围：`D:\PycharmProjects\Python Practice\Movie Comments`  
清理前：约 34.748 GiB / 20,342 个文件  
清理后：约 1.91 GiB / 73 个文件  
实际释放：约 32.84 GiB

> 已根据用户本次明确授权执行清理。以下内容作为保留边界、删除范围和验收记录留存。

## 一、必须保留：界面运行源码

- `streamlit_app.py`
- `best_sentiment_ensemble_v2.py`
- `sentiment_utils.py`
- `reputation_analysis.py`
- `prepare_sentiment_v2.py`
- `feature_engineering.py`
- `README.md`（建议保留；内容部分过时，但仍是项目说明）

运行依赖链：

`streamlit_app.py -> sentiment_utils.py -> prepare_sentiment_v2.py -> feature_engineering.py`

口碑图表依赖 `reputation_analysis.py`，最佳模型推理依赖 `best_sentiment_ensemble_v2.py`。

## 二、必须保留：当前图表数据与封存数据

- `assets/data/douban_comments.csv`
- `outputs/sentiment/bert-v2/splits/train.csv`
- `outputs/sentiment/bert-v2/splits/valid.csv`
- `outputs/sentiment/bert-v2/splits/test.csv`（界面不会读取；为最终验收继续封存）
- `outputs/sentiment/bert-v2/manifest.json`（建议保留；记录输入哈希、固定拆分和数据血缘）

当前图表只使用固定 `train + valid` 的 1,816 条数据；`test.csv` 没有被审计或界面读取。

## 三、必须保留：最佳三分类集成的 8 个目录

以下目录均位于 `outputs/sentiment/bert-v2/experiments/`，建议整目录保留，共约 1.909 GiB：

- `ensemble_diverse_weighted_calibrated_v2`（当前最佳顶层集成）
- `ensemble_embedding_rating5_baseline_calibrated_v1`（嵌套集成）
- `frozen_embedding_clspmean_bestencoder_seed42`（冻结嵌入分类器）
- `rating5mix_to_canonical3_ce_lr1e5_seed42`（冻结嵌入编码器）
- `rating5_to_canonical3_ce_seed42`
- `baseline_roberta_ce_seed2026`
- `rating5mix500_to_canonical3_ce_lr1e5_seed42`
- `cumbce_to_canonical3_ce_lr1e5_seed42`

这 8 个目录包含运行所需的 `metrics.json`、`classifier.pkl`、`best_model/`，以及体积很小但重要的验证报告、预测结果和训练记录。

## 四、已清理：缓存、日志、旧图表与空目录

这些路径不参与当前 Streamlit 最佳模型运行：

- `.npm-cache/`（约 751.1 MiB）
- `.uv_cache/`（约 18.1 MiB）
- `__pycache__/`（约 0.7 MiB）
- `assets/pictures/`（约 1.1 MiB；当前图表实时生成）
- `outputs/runtime_logs/`
- `outputs/streamlit_ui.stdout.log`
- `outputs/streamlit_ui.stderr.log`
- `outputs/screenshots/`
- `outputs/checks/`
- `outputs/performance_comparison/`
- `docs/`（旧训练参数文档，非运行依赖）

## 五、已清理：未使用的数据和本地基础模型

当前界面和最佳集成不读取以下路径：

- `assets/models/`（约 10.719 GiB；训练基础模型缓存，删除后需重新下载才能从头复现）
- `assets/data/external/`（约 543.3 MiB）
- `assets/data/cleaned_data.csv`（旧全量图表数据）
- `assets/data/crawl_test.csv`
- `assets/data/movies.csv`
- `outputs/sentiment/bert-v2/pretraining/`（约 390.7 MiB）
- `outputs/sentiment/bert-v2/data/`（约 46.0 MiB；离线训练中间数据）
- `outputs/sentiment/bert-v2/rating5/`（约 13.9 MiB）
- `outputs/sentiment/bert-v2/ovr/`（约 3.6 MiB）
- `outputs/sentiment/bert-v2/hierarchical/`（约 3.0 MiB）
- `outputs/sentiment/bert-v2/ensemble_equal_weight_audit.json`
- `outputs/sentiment/bert-v2/ensemble_equal_weight_embedding_audit.json`
- `outputs/sentiment/bert-v2/ensemble_equal_weight_rdrop_audit.json`
- `outputs/sentiment/bert-v2/ensemble_equal_weight_seeddiversity_audit.json`
- `outputs/sentiment/bert-v2/erlangshen330m_ensemble_probe.json`
- `outputs/sentiment/bert-v2/macbert_ensemble_probe.json`
- `outputs/sentiment/bert-v2/experiment_summary.csv`
- `outputs/sentiment/bert-v2/experiment_summary.json`

风险说明：删除本节内容不会影响当前界面推理，但会降低重新训练、复现实验或离线分析的便利性。

## 六、已清理：64 个未被最佳集成引用的实验目录

以下目录均位于 `outputs/sentiment/bert-v2/experiments/`，合计约 20.386 GiB。

### 6.1 训练血缘模型（约 1.526 GiB）

不影响当前推理；删除后无法直接从这些阶段继续训练。

- `rating5_dmsc50k_ordinal_seed42`
- `rating5_mix1000pc_cumbce_seed42`
- `rating5_mix1000pc_ordinal_seed42`
- `rating5_mix500pc_ordinal_seed42`

### 6.2 两个超大未采用模型（约 2.427 GiB）

- `erlangshen_330m_sentiment_to_canonical3_ce_seed42`
- `roberta_large_ce_llrd095_seed42`

### 6.3 其他常规模型实验（约 16.397 GiB）

- `baseline_roberta_ce_seed13`
- `baseline_roberta_ce_seed42`
- `canonical_ce_ls005_sqrtweights_seed42`
- `canonical_ordinalce_l05_ls005_seed42`
- `dapt_dmsc30k_to_canonical_ce_lr2e5_seed42`
- `dapt_to_rating5_mix1000pc_ordinal_seed42`
- `deberta97m_ce_seed42`
- `erlangshen_headinit_to_canonical3_ce_seed42`
- `erlangshen_sentiment_to_canonical3_ce_seed42`
- `hier_negative_neutral_sqrt_seed42`
- `hier_positive_seed42`
- `m3e_ce_lr1e5_seed42`
- `macbert_base_ce_seed42`
- `minimaltext_roberta_ce_seed42`
- `mix_crawler_targeted_ce_seed42`
- `mix_external1500pc_ce_ls005_seed42`
- `mix_multisent_neu1000_other500_ce_seed42`
- `moviepair_roberta_ce_seed42`
- `multitask_aux03_seed42`
- `negative_ovr_sqrt_seed42`
- `neutral_ovr_sqrt_seed42`
- `pooling_cls_mean_max_seed42`
- `rating5_merged_head_3class`
- `rating5_to_canonical3_ce_lr2e5_seed42`
- `rating5_to_canonical3_ce_ls005_lr1e5_seed42`
- `rating5merged_to_canonical3_ce_seed42`
- `rating5mix_to_canonical3_ce_lr1e5_seed13`
- `rating5mix_to_canonical3_ce_lr1e5_seed17`
- `rating5mix_to_canonical3_ce_lr1e5_seed2026`
- `rating5mix_to_canonical3_fgm1_seed42`
- `rating5mix_to_canonical3_focal_g1_seed42`
- `rating5mix_to_canonical3_freeze6_ce_seed42`
- `rating5mix_to_canonical3_gce_q07_seed42`
- `rating5mix_to_canonical3_llrd09_retry1_seed42`
- `rating5mix_to_canonical3_rdrop1_seed42`
- `rating5mix_to_canonical3_sam005_seed42`
- `rating5mix_to_canonical3_starsoft_moderate_seed42`
- `rating5mix_to_canonical3_supcon005_t01_seed42`
- `rating5mix_to_canonical3_tokenmask005_seed42`
- `smoke_20260710_1736`
- `stage1_dmsc150k_ce_ls005_seed42`
- `stage2_dmsc_to_canonical_ce_ls005_lr1e5_seed42`
- `teacherfiltered_2k4k2k_continue_seed42`

### 6.4 小型、空实验与后处理结果（约 36.6 MiB）

- `embedding_rbf_bestencoder_seed42`
- `frozen_embedding_clspmean_rating5encoder_seed42`
- `frozen_embedding_mean_m3e_seed42`
- `hier_product_sqrt_seed42`
- `hier_product_sqrt_seed42_v2`
- `hybrid_embedding_char_movie_surface_retry1_seed42`
- `hybrid_embedding_char_movie_surface_seed42`
- `qwen25_05b_lora_r8_fast_seed42`
- `qwen25_05b_lora_r8_seed42`
- `rating5mix_to_canonical3_llrd09_seed42`
- `rating5mix_to_canonical3_mc_dropout20_seed42`
- `smoke_inputs_20260710_1736`
- `stacking_oof_best3_v1`
- `stacking_oof_diverse4_v1`
- `tfidf_char_grid_seed42`

## 七、已清理：非运行时实验脚本

以下顶层脚本不被当前 Streamlit 入口导入：

- `build_hierarchical_data.py`
- `build_movie_pair_variant.py`
- `build_oof_noise_weights_v2.py`
- `build_ovr_sentiment_data_v2.py`
- `build_sentiment_mix.py`
- `build_text_variants.py`
- `continue_mlm_pretraining_v2.py`
- `convert_binary_sentiment_checkpoint.py`
- `convert_rating5_checkpoint.py`
- `crawler.py`
- `error_analysis.py`
- `evaluate_fixed_ensembles_v2.py`
- `evaluate_hierarchical_sentiment_v2.py`
- `evaluate_mc_dropout_sentiment_v2.py`
- `evaluate_oof_stacking_v2.py`
- `materialize_calibrated_ensemble_v2.py`
- `materialize_weighted_ensemble_v2.py`
- `prepare_crawler_increment_v2.py`
- `prepare_multilingual_sentiment_mix_v2.py`
- `prepare_rating5_v2.py`
- `prepare_star_soft_labels_v2.py`
- `select_confident_external_v2.py`
- `summarize_sentiment_experiments_v2.py`
- `train_bert_sentiment.py`
- `train_bert_sentiment_v2.py`
- `train_embedding_classifier_v2.py`
- `train_hybrid_embedding_text_classifier_v2.py`
- `train_ml_sentiment.py`
- `train_multitask_sentiment_v2.py`
- `train_pooling_sentiment_v2.py`
- `train_qwen_lora_sentiment_v2.py`
- `train_tfidf_sentiment_v2.py`
- `validate_project_outputs.py`

风险说明：这些文件体积很小。删除不会影响当前界面，但会丢失训练与实验复现代码。若仍要继续冲击 Macro F1 > 0.8，建议暂时保留 `train_bert_sentiment_v2.py`、相关数据准备/评估脚本以及本清单，等目标完成后再清理。

## 八、清理后验收条件

清理完成后已验证：

1. `python -m py_compile` 能编译六个运行源码文件。
2. `BestSentimentEnsemble(device="cpu")` 能加载并预测三分类概率。
3. `load_best_model_dashboard_data()` 返回 1,816 行，且只含 train/valid。
4. `http://localhost:8501/` 返回 200。
5. 页面不显示“最佳集成加载失败”，预测结果标为“最佳五模型加权集成”。
