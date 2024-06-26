'''
Generate fake audios from AudioGen
'''

import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate 5 seconds.

descriptions = [
    "The bustling sounds of a metropolitan subway station during rush hour.",
    "Leaves rustling as a soft autumn breeze sweeps through a forest.",
    "The distant roar of a waterfall cascading into a crystal-clear pool.",
    "City park ambience with children playing and birds chirping in the background.",
    "The soft patter of rain on a tin roof during a gentle shower.",
    "A campfire crackling fiercely on a cold winter's night in the wilderness.",
    "Busy office sounds with typing, phone ringing, and low chatter among colleagues.",
    "A thunderstorm approaching, with thunder rumbling and lightning flashing.",
    "The serene silence of a snowy landscape, interrupted only by the crunch of footsteps.",
    "A street musician playing violin, surrounded by the sounds of a bustling city.",
    "The rhythmic chugging and whistle of an old steam train starting its journey.",
    "Coffee shop ambiance with the sound of espresso machines and muffled conversations.",
    "A busy kitchen environment with chefs calling orders and the sizzle of frying.",
    "Frogs croaking rhythmically in a marshy pond at dusk.",
    "A bee buzzing around a blooming garden full of flowers and life.",
    "A classroom filled with the sound of students whispering and pens scribbling.",
    "The echoing call of a loon across a misty lake in the early morning.",
    "Wind howling through a narrow alleyway in a deserted town.",
    "A bustling fish market with vendors shouting and ice being shuffled.",
    "A horse-drawn carriage rolling over cobblestones in an old European city.",
    "A mountain stream trickling over rocks, surrounded by the silence of nature.",
    "A busy pedestrian street with footsteps, laughter, and the occasional bike bell.",
    "A lone wolf howling under a full moon in a dense forest.",
    "The clink of glasses and soft music at an outdoor cafe.",
    "The sound of a pencil sketching rapidly on a piece of paper.",
    "A construction site with hammering, drilling, and shouts of workers.",
    "Chimes tinkling softly in a gentle breeze outside a countryside cottage.",
    "The distant sound of a church bell ringing through a quiet village.",
    "An orchestra tuning their instruments before a concert begins.",
    "A farmer's market with vendors chatting and the rustle of fresh produce.",
    "The steady drip of a cave's stalactites into an underground pool.",
    "An art gallery with the soft footsteps of patrons moving from piece to piece.",
    "The electric buzz of a neon sign in a quiet street at night.",
    "A carnival with the sounds of rides, games, and excited visitors.",
    "The solemn tolling of a bell in a distant lighthouse during fog.",
    "A bustling newsroom with the clack of keyboards and urgent conversations.",
    "The vibrant sound of a mariachi band performing at an outdoor festival.",
    "Children laughing and splashing in a community swimming pool.",
    "The eerie silence of an abandoned building, with occasional creaks.",
    "A lively bar with the clatter of billiard balls and background music.",
    "An underground cave with dripping water and the flutter of bat wings.",
    "A tailor's shop with the hum of sewing machines and snip of scissors.",
    "The muffled sound of a parade heard from inside an apartment.",
    "A greenhouse full of plants with the subtle sound of misting water.",
    "The crisp sound of a golf club hitting a ball on a quiet morning.",
    "A dog park with barking dogs, owners chatting, and the jingle of collars.",
    "The sound of a typewriter clicking rapidly in a quiet office.",
    "The bustling of a dock with boats bumping and seagulls squawking.",
    "A high school hallway with lockers slamming and the buzz of students.",
    "The soft gurgle of a brook winding through a meadow filled with wildflowers.",
    "A thunderstorm approaching, with rolling thunder and flashes of lightning.",
    "The quiet hum of a library filled with students studying and pages turning.",
    "Coffee shop atmosphere with the sound of espresso machines and murmur of conversations.",
    "The rhythmic clacking of a train on tracks, traveling through the countryside.",
    "A campfire crackling fiercely on a cold, starry night in the wilderness.",
    "Frogs croaking rhythmically near a swamp in the twilight hours.",
    "Footsteps crunching on a gravel path during a serene hike in the mountains.",
    "The sound of a pencil sketching on paper in a quiet artist's studio.",
    "The sound of a bustling fish market, with vendors shouting and fish splashing.",
    "Wind chimes gently tinkling in a soft breeze on a sunny afternoon.",
    "The sound of a jet ski cutting through waves on a busy lake.",
    "Ice cracking underfoot while walking on a frozen pond in winter.",
    "The mechanical noise of a wind turbine generating power on a windy day.",
    "The distant sound of a foghorn warning ships on a foggy morning at sea.",
    "Children laughing and splashing in a public swimming pool.",
    "The eerie silence of an abandoned building, with occasional drips of water.",
    "The sizzle of meat grilling on a barbecue during a summer cookout.",
    "The sound of typing on a mechanical keyboard in a tech office.",
    "The gentle snoring of a sleeping dog curled up in its bed.",
    "A clock ticking steadily in an otherwise silent room.",
    "The sound of a crowd cheering at a concert when the band takes the stage.",
    "A chainsaw buzzing as it cuts through a log in the forest.",
    "The beep of a microwave as it finishes cooking.",
    "A rooster crowing at the break of dawn in a rural farmyard.",
    "The squeal of brakes as a bus comes to a stop at a city bus stop.",
    "The clinking of glasses and laughter at a lively dinner party.",
    "The sound of a helicopter flying low over a city.",
    "A cat purring contentedly while being petted.",
    "The sound of waves crashing against a lighthouse during a storm.",
    "An owl hooting in a forest at night.",
    "The beep of a car locking as someone uses a remote key.",
    "The sound of a fan blowing on a hot day in a small room.",
    "The buzz of a mosquito flying close to the ear in a dark room.",
    "The jingle of keys as someone walks briskly.",
    "The sound of a river flowing rapidly over rocks.",
    "An air conditioner humming in the background on a hot day.",
    "The sound of popcorn popping in a machine at the cinema.",
    "The sound of a guitar being tuned by a musician before a performance.",
    "A door creaking open slowly in an old house.",
    "The sound of scissors cutting through paper.",
    "A printer whirring as it prints out a document.",
    "The sound of a stapler clicking as pages are stapled together.",
    "A kettle whistling as water reaches boiling point.",
    "A cat meowing for food in the morning.",
    "The sound of a basketball bouncing on an outdoor court.",
    "The gurgle of a stream trickling through a forest.",
    "A balloon popping suddenly at a party.",
    "The ding of an elevator arriving at a floor.",
    "The sound of a spray paint can being shaken and then sprayed on a wall.",
    "A cricket chirping on a quiet summer night."]

for i in range(len(descriptions)):
    wav = model.generate(descriptions=[descriptions[i]], progress=True) 

    # for idx, one_wav in enumerate(wav):
    #     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'./data/AudioGen/{i}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)