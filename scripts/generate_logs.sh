# Generate event logs
uv run python -m generator.gen_anomalous_eventlog_syn

wait
echo "Event logs generated successfully."