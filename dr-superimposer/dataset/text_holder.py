class TextHolder:
    texts = []

    def register(self, text):
        id = len(self.texts)
        self.texts.append(text)
        return id

    def get(self, id):
        return self.texts[id]
