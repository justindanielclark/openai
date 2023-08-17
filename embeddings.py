import faiss
from sentence_transformers import SentenceTransformer
sentences = [
    "Dolphins are known for their high intelligence and playful behavior.",
    "The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph (112 km/h).",
    "Bees perform a crucial role in pollinating plants, which helps in food production.",
    "Elephants have the largest brain of any land animal and display complex social behaviors.",
    "The blue whale is the largest animal to have ever existed on Earth.",
    "Some species of octopus can change their skin color and texture to blend in with their surroundings.",
    "The mimic octopus can imitate the appearance and behavior of various marine creatures.",
    "Bats are the only mammals capable of sustained flight.",
    "Camels are well adapted to desert environments, with the ability to store water and tolerate high temperatures.",
    "Penguins are excellent swimmers and can dive to great depths in search of food.",
    "Male seahorses are the ones who become pregnant and give birth to their offspring.",
    "The monarch butterfly is known for its incredible migration, covering thousands of miles.",
    "The intelligence of ravens and crows is comparable to that of some primates.",
    "The honey badger is known for its fearless nature and strong defensive abilities.",
    "Chameleons have the ability to independently move their eyes, allowing them to look in different directions simultaneously.",
    "The platypus is a unique mammal that lays eggs and produces milk but lacks nipples.",
    "The komodo dragon is the world's largest lizard and has a venomous bite.",
    "Koalas primarily feed on eucalyptus leaves, which are toxic to most other animals.",
    "The humpback whale is known for its elaborate songs, which can travel long distances underwater.",
    "Ostriches are the largest and fastest-running birds, with the ability to reach speeds of up to 45 mph (72 km/h).",
    "Many species of fireflies use their bioluminescence to communicate and attract mates.",
    "The narwhal is often referred to as the \"unicorn of the sea\" due to its long, spiral tusk.",
    "Male lions are recognizable by their majestic manes, which indicate their age and health.",
    "The red panda, despite its name, is not closely related to the giant panda and has its own unique lineage.",
    "Some species of frogs can \"freeze\" during winter months and thaw out when temperatures rise.",
    "The hummingbird is the only bird capable of sustained hovering flight.",
    "The Tasmanian devil is a carnivorous marsupial known for its loud and aggressive vocalizations.",
    "Emperor penguins endure harsh Antarctic conditions, with males incubating eggs on their feet.",
    "Axolotls are salamanders that retain their aquatic juvenile form throughout their lives.",
    "The great white shark is one of the most well-known and feared predators of the ocean."
]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
dimensions = embeddings.shape[1]
index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)


query = ["Give me uninteresting animal facts"]
xq = model.encode(query)
k = 30

D, I = index.search(xq, k)

for i in I[0]:
    print(sentences[i])