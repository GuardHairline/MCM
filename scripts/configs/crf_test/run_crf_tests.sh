#!/bin/bash
# CRF修复测试批量运行脚本

echo '=========================================='
echo 'CRF修复测试 - 批量运行'
echo '=========================================='
echo '总实验数: 12'
echo ''

echo '[1/12] 运行任务: MATE'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/baseline_twitter2015_mate.json
if [ $? -ne 0 ]; then
    echo '✗ mate 失败'
else
    echo '✓ mate 完成'
fi
echo ''

echo '[2/12] 运行任务: MATE'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_only_twitter2015_mate.json
if [ $? -ne 0 ]; then
    echo '✗ mate 失败'
else
    echo '✓ mate 完成'
fi
echo ''

echo '[3/12] 运行任务: MATE'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/span_only_twitter2015_mate.json
if [ $? -ne 0 ]; then
    echo '✗ mate 失败'
else
    echo '✓ mate 完成'
fi
echo ''

echo '[4/12] 运行任务: MATE'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_and_span_twitter2015_mate.json
if [ $? -ne 0 ]; then
    echo '✗ mate 失败'
else
    echo '✓ mate 完成'
fi
echo ''

echo '[5/12] 运行任务: MNER'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/baseline_twitter2015_mner.json
if [ $? -ne 0 ]; then
    echo '✗ mner 失败'
else
    echo '✓ mner 完成'
fi
echo ''

echo '[6/12] 运行任务: MNER'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_only_twitter2015_mner.json
if [ $? -ne 0 ]; then
    echo '✗ mner 失败'
else
    echo '✓ mner 完成'
fi
echo ''

echo '[7/12] 运行任务: MNER'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/span_only_twitter2015_mner.json
if [ $? -ne 0 ]; then
    echo '✗ mner 失败'
else
    echo '✓ mner 完成'
fi
echo ''

echo '[8/12] 运行任务: MNER'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_and_span_twitter2015_mner.json
if [ $? -ne 0 ]; then
    echo '✗ mner 失败'
else
    echo '✓ mner 完成'
fi
echo ''

echo '[9/12] 运行任务: MABSA'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/baseline_twitter2015_mabsa.json
if [ $? -ne 0 ]; then
    echo '✗ mabsa 失败'
else
    echo '✓ mabsa 完成'
fi
echo ''

echo '[10/12] 运行任务: MABSA'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_only_twitter2015_mabsa.json
if [ $? -ne 0 ]; then
    echo '✗ mabsa 失败'
else
    echo '✓ mabsa 完成'
fi
echo ''

echo '[11/12] 运行任务: MABSA'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/span_only_twitter2015_mabsa.json
if [ $? -ne 0 ]; then
    echo '✗ mabsa 失败'
else
    echo '✓ mabsa 完成'
fi
echo ''

echo '[12/12] 运行任务: MABSA'
python -m scripts.train_with_zero_shot --config scripts/configs/crf_test/crf_and_span_twitter2015_mabsa.json
if [ $? -ne 0 ]; then
    echo '✗ mabsa 失败'
else
    echo '✓ mabsa 完成'
fi
echo ''

echo '=========================================='
echo '所有测试完成!'
echo '=========================================='
