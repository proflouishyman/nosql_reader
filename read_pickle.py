import pickle

# Specify the filename
filename = 'unique_terms.pkl'

try:
    # Load the contents of the .pkl file
    with open(filename, 'rb') as f:
        unique_terms = pickle.load(f)
    
    # Display the contents
    print("Contents of unique_terms.pkl:")
    for field, terms in unique_terms.items():
        print(f"\nField: {field}")
        print("Words:")
        for word, count in terms['words'].items():
            print(f"  {word}: {count}")
        print("Phrases:")
        for phrase, count in terms['phrases'].items():
            print(f"  {phrase}: {count}")

except Exception as e:
    print(f"Error loading {filename}: {e}")
