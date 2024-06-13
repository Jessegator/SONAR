import os
import torch
import json
from TTS.api import TTS


root_path = './LibriTTS_ref_audios'
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

prompts = [
    "Today's weather forecast predicts a slight chance of rain with intermittent sunny spells in the late afternoon.",
    "The quick brown fox jumps over the lazy dog while the sun sets behind the rolling hills.",
    "She opened the last book she bought and found a handwritten note tucked between the pages.",
    "While exploring ancient ruins, a group of archaeologists stumbled upon a hidden tomb sealed for centuries.",
    "Can you believe it? She said that they might go to the moon for their honeymoon next year!",
    "In a dramatic turn of events, the underdog team won the championship against all odds last night.",
    "Financial markets fluctuated wildly this week, with tech stocks leading the charge both up and down.",
    "The soft hum of the city at night has a rhythm and a pulse like a living creature.",
    "Farmers in the coastal regions are adopting new, sustainable practices to combat the effects of climate change.",
    "How often do you travel abroad? This year, I plan to visit at least three new countries.",
    "He plays the violin with such passion that even the most disinterested person would stop to listen.",
    "Artificial intelligence is rapidly changing the way we interact with technology on a daily basis.",
    "The recipe calls for a pinch of salt, a teaspoon of sugar, and a whole lot of love.",
    "Why do we dream when we sleep? Scientists continue to explore the mysteries of the human mind.",
    "Lights flickered as the storm intensified, casting eerie shadows across the deserted street.",
    "Local museums are collaborating on an exhibit that showcases the region's history and cultural heritage.",
    "The investment firm has announced a new strategy aimed at boosting long-term growth and sustainability.",
    "In the quiet of the forest, you can hear the whisper of the trees telling ancient stories.",
    "Fashion trends this season include bold colors, geometric patterns, and a nod to retro styles.",
    "He joked about the situation, but I could tell he was worried by the tone of his voice.",
    "The concert, featuring classical and contemporary pieces, will be held at the city hall this Saturday.",
    "Voters are heading to the polls today to decide on a range of important local and national issues.",
    "Medical research has led to a breakthrough in the treatment of a rare, previously untreatable condition.",
    "She taught her dog to fetch the newspaper every morning, rain or shine.",
    "Engineers are developing a new type of battery that could significantly extend the life of electronic devices.",
    "This documentary explores the lives of nomadic tribes in the desert and their adaptations to the environment.",
    "The board game, designed to test strategic thinking, has become a favorite at family gatherings.",
    "After years of study, the young scientist was ready to present her findings to the world.",
    "A gentle breeze carried the scent of jasmine through the open windows of the quiet summer home.",
    "The chef explained that the secret to perfect pasta is in the timing of the sauce.",
    "Adventurers recount their thrilling experience in the icy tundra and the lessons they learned about survival.",
    "Efforts to preserve local languages are gaining momentum, with communities leading the initiative.",
    "The novel is set in a dystopian future where society is divided by technological access.",
    "Last night's debate highlighted the candidates' differing approaches to major economic policies.",
    "He repaired old clocks, a craft he learned from his grandfather, with patience and precision.",
    "The sculpture, made entirely from recycled materials, won first prize at the art festival.",
    "To master chess, one must not only understand the pieces' movements but also anticipate the opponent's strategy.",
    "A new study suggests that listening to music can improve memory retention and cognitive function.",
    "She reminisced about her childhood adventures, exploring the woods and streams near her home.",
    "Drivers are advised to take alternate routes due to the road closure this weekend for the annual parade.",
    "The new science fiction series blends elements of adventure and philosophy, set in a parallel universe.",
    "Tourists flock to the small island every year to experience its pristine beaches and vibrant local culture.",
    "The bakery is famous for its homemade bread, which uses a traditional recipe passed down through generations.",
    "The debate on the impact of social media on youth mental health continues to grow.",
    "He crafted each piece of the model ship with meticulous detail, ensuring every part was historically accurate.",
    "Her painting won the award for its innovative use of colors and textures, capturing the viewer's imagination.",
    "Experts discuss the potential of renewable energy sources to completely replace fossil fuels in the future.",
    "A sudden inspiration struck her, and she began to write a poem that perfectly expressed her feelings.",
    "The film festival showcases independent films that challenge traditional narratives and provoke thought.",
    "Participants in the marathon ran through the city's historic districts, cheered on by crowds of supporters.",
    "The old library is home to rare manuscripts dating back hundreds of years, each with its own story.",
    "Understanding the complexities of the human brain is one of the greatest challenges in modern science.",
    "Legends tell of a mysterious artifact hidden deep within the ancient forest, guarded by creatures unknown to mankind.",
    "The recent volcanic eruption has led to a drastic change in the landscape, affecting local wildlife and ecosystems.",
    "Experts predict that the next decade will bring significant advances in quantum computing, altering the tech landscape.",
    "She scanned the horizon, her eyes catching the first light of dawn, signaling a new day full of possibilities.",
    "In a surprising turn of events, the quiet town became a hub for international artists and creatives.",
    "The best way to understand the universe is to look up at the stars on a clear night.",
    "To make the perfect omelette, you need patience and a little bit of skill to flip it just right.",
    "Economic analysts are debating the potential impacts of the new trade policy on global markets.",
    "A rare blue moon will be visible tonight, an event that occurs only once every few years.",
    "His novel, set in the early 1900s, captures the essence of life in a small European village.",
    "With each step up the mountain, the air grew thinner, but her determination did not waver.",
    "The film director's unique vision brings a fresh perspective to the classic genre of noir cinema.",
    "The annual tech conference will feature innovations in artificial intelligence and virtual reality this year.",
    "Philosophers and scientists continue to debate the concept of free will and its implications for morality.",
    "This week's cooking challenge is to create a dish using only locally sourced, sustainable ingredients.",
    "The jazz band's smooth tunes fill the air, providing a perfect soundtrack for the evening.",
    "Wildfires continue to spread across the region, prompting emergency evacuations and major rescue operations.",
    "Astronauts share their experiences of seeing Earth from space, describing it as a profoundly moving moment.",
    "The old man told stories of the sea, his voice echoing the waves and the winds of past storms.",
    "Fitness experts recommend mixing cardio with strength training for a balanced exercise routine.",
    "The city's ancient walls, dating back to the medieval period, tell tales of history and conquest.",
    "New archaeological findings suggest that the area was inhabited much earlier than previously thought.",
    "The debate over the ethical implications of cloning continues to divide scientists and the general public.",
    "She designs fashion that defies trends, focusing instead on timeless elegance and sustainability.",
    "Tonight's lecture will explore the intersection of technology and education, and its impact on future learning.",
    "The community garden not only provides fresh produce but also a sense of connection among the neighbors.",
    "In response to rising sea levels, architects are designing floating houses that can adapt to changing tides.",
    "The photographer's exhibition captures the vibrant life of street markets around the world.",
    "As the orchestra tuned their instruments, the anticipation in the concert hall built to a palpable level.",
    "Experts discuss the future of transportation, predicting that autonomous vehicles will become commonplace by 2030.",
    "The mystery novel, set in a remote village, twists and turns through secrets buried deep in the snow.",
    "The documentary contrasts the lives of people living in megacities with those in remote rural areas.",
    "Understanding the role of microorganisms in our ecosystem can help us appreciate the complexity of life.",
    "The chef's new fusion cuisine blends flavors from around the world, creating surprising and delightful dishes.",
    "The debate team argued about the effectiveness of current cybersecurity measures in protecting personal data.",
    "In her speech, the activist emphasized the urgency of addressing climate change to protect future generations.",
    "The theater group's latest production features a revolutionary stage design that brings the play to life.",
    "His poetry captures the essence of urban life, with its chaos and moments of unexpected beauty.",
    "The science fair projects this year range from robotics to environmental conservation efforts.",
    "She reminisced about the summer days spent by the lake, the water shimmering under the starry night sky.",
    "Recent studies highlight the importance of mental health days and their positive impact on workplace productivity.",
    "The guitar solo, rich with emotion, captivated the audience, leaving them cheering for more.",
    "The seminar on digital privacy provided attendees with practical tips on securing online information.",
    "During the workshop, participants learned how to repair and upcycle old furniture, giving it new life.",
    "The café serves a blend of coffee known for its robust flavor and aromatic properties.",
    "The painter's latest series, inspired by nature's forms and colors, has gained critical acclaim.",
    "Advances in biotechnology are paving the way for more effective treatments for genetic diseases.",
    "The farmer's market on Saturday offers fresh fruits, vegetables, and local crafts to the community."]

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


reference_audios = os.listdir('./LibriTTS_ref_audios')

assert len(prompts) == len(reference_audios)


for i in range(len(prompts)):

    audio_path = os.path.join(root_path, reference_audios[i])


    # Run TTS
    # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
    # Text to speech to a file

    tts.tts_to_file(text=prompts[i], speaker_wav=audio_path, language="en",file_path="./xtts/{}".format(reference_audios[i].split('_')[0]+'_0.wav'))




