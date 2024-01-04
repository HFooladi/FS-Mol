import pickle


def main():
    with open("embedding.p", "wb") as f:
        pickle.dump(embedding, f)


if __name__ == "__main__":
    main()
