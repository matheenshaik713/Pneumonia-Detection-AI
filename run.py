# from src.model import PneumoniaCNN

# model = PneumoniaCNN.build()

# model.summary()
# from src.callbacks import get_callbacks

# callbacks = get_callbacks()

# print(f"Number of callbacks: {len(callbacks)}")

# for callback in callbacks:
#     print(type(callback).__name__)

# from src.trainer import Trainer

# trainer = Trainer()

# trainer.compile()

# trainer.model.summary()

# history = trainer.train()

# trainer.save()
# from src.dataloader import DataLoader

# loader = DataLoader()

# train_ds = loader.get_train_dataset()

# val_ds = loader.get_validation_dataset()

# test_ds = loader.get_test_dataset()

# print()

# print("Classes :", loader.get_class_names())

# print("Train batches :", len(train_ds))

# print("Validation batches :", len(val_ds))

# print("Test batches :", len(test_ds))

# from src.trainer import Trainer

# trainer = Trainer()

# trainer.run()
# from src.evaluate import Evaluator

# evaluator = Evaluator()

# evaluator.evaluate()

# from src.plots import PlotGenerator

# plots = PlotGenerator()

# plots.generate()


# from src.predict import Predictor

# predictor = Predictor()

# predictor.predict(
#     "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
# )
