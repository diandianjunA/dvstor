# Deep100M 实验

该目录预留给 Deep100M 的三项 Motivation 实验和消融实验。

Deep100M 的向量类型、维度、原始数据路径、查询集、Ground Truth 和索引前缀尚未在
当前仓库中确认，因此这里不复制 SIFT100M 的运行脚本，避免产生看似可运行、实际读取
错误数据或索引的配置。完成 Deep100M 数据与索引接入后，应在本目录提供与
`../sift100m/` 相同的四个入口：`program1`、`program2`、`program3` 和 `ablation`。
