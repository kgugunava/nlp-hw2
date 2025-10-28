from translator import Translator

def main():
    translator = Translator('input.txt', 'output.txt')
    translator.start()

if __name__ == "__main__":
    main()