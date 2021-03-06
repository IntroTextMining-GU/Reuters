from nltk.corpus import reuters

def loadData():
    doc_id = reuters.fileids()
    train_id = [d for d in doc_id if d.startswith('training/')]
    test_id = [d for d in doc_id if d.startswith('test/')]

    train_data = [reuters.raw(doc_id) for doc_id in train_id]
    test_data = [reuters.raw(doc_id) for doc_id in test_id]

    train_label = [reuters.categories(doc_id) for doc_id in train_id]
    test_label = [reuters.categories(doc_id) for doc_id in test_id]

    return {"train": train_data, "test": test_data}, {"train": train_label, "test": test_label}
