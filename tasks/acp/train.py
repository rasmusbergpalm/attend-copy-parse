import trainer
from tasks.acp.acp import AttendCopyParse

print("Training...")
model = trainer.train(AttendCopyParse())

print("Testing...")
model.test_set()
