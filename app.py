import gradio as gr
import asyncio
import json
import functools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, function_tool

movie_knowledge_base = [
    {
        "title": "A Goofy Movie",
        "description": "Max Goof, a teenager, embarks on a chaotic road trip with his bumbling father, Goofy, to bond and attend a concert, navigating teenage rebellion and family love.",
        "genre": ["Animation", "Children", "Comedy", "Romance"],
        "director": "Kevin Lima",
        "actors": ["Bill Farmer", "Jason Marsden", "Jim Cummings"],
        "year": 1995,
        "box_office": 35348597,
        "budget": 18000000,
        "awards": [],
        "rating": 4.5
    },
    {
        "title": "Gumby: The Movie",
        "description": "Gumby, a claymation character, and his friends must stop the evil Blockheads from replacing their town with robots during a benefit concert.",
        "genre": ["Animation", "Children"],
        "director": "Art Clokey",
        "actors": ["Dal McKennon", "Art Clokey", "Gloria Clokey"],
        "year": 1995,
        "box_office": 57076,
        "budget": 2800000,
        "awards": [],
        "rating": 3.5
    },
    {
        "title": "The Swan Princess",
        "description": "A princess cursed to become a swan must rely on true love to break the spell, facing an evil sorcerer in this animated fairy tale.",
        "genre": ["Animation", "Children"],
        "director": "Richard Rich",
        "actors": ["Michelle Nicastro", "Howard McGillin", "Jack Palance"],
        "year": 1994,
        "box_office": 9771650,
        "budget": 45000000,
        "awards": [],
        "rating": 3.0
    },
    {
        "title": "The Lion King",
        "description": "Simba, a young lion prince, flees after his father’s death but returns to reclaim his throne from his treacherous uncle Scar in this animated epic.",
        "genre": ["Adventure", "Animation", "Children", "Drama", "Musical", "IMAX"],
        "director": ["Roger Allers", "Rob Minkoff"],
        "actors": ["Matthew Broderick", "Jeremy Irons", "James Earl Jones"],
        "year": 1994,
        "box_office": 968483777,
        "budget": 45000000,
        "awards": ["Oscar Best Original Score", "Oscar Best Original Song", "Golden Globe Best Motion Picture - Musical or Comedy"],
        "rating": 4.5
    },
    {
        "title": "The Secret Adventures of Tom Thumb",
        "description": "A tiny boy, born to normal parents, is kidnapped and navigates a dark, surreal world of science and survival in this stop-motion tale.",
        "genre": ["Adventure", "Animation"],
        "director": "Dave Borthwick",
        "actors": ["Nick Upton", "Deborah Collard", "Frank Passingham"],
        "year": 1993,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 2.5
    },
    {
        "title": "This So-Called Disaster",
        "description": "A documentary capturing actor Sam Shepard as he directs a play, blending rehearsals with personal reflections on art and life.",
        "genre": ["Documentary", "Disaster"],
        "director": "Michael Almereyda",
        "actors": ["Sam Shepard", "Nick Nolte", "Sean Penn"],
        "year": 2003,
        "box_office": 46658,
        "budget": 0,
        "awards": [],
        "rating": 3.0
    },
    {
        "title": "Love and Other Disasters",
        "description": "A fashion magazine assistant in London juggles romance, friendship, and chaos while trying to matchmake those around her.",
        "genre": ["Comedy", "Romance", "Disaster"],
        "director": "Alek Keshishian",
        "actors": ["Brittany Murphy", "Matthew Rhys", "Santiago Cabrera"],
        "year": 2006,
        "box_office": 6743917,
        "budget": 10000000,
        "awards": [],
        "rating": 3.0
    },
    {
        "title": "Disaster Movie",
        "description": "A group of friends faces absurd, over-the-top disasters in a parody of blockbuster films, filled with pop culture references.",
        "genre": ["Comedy", "Disaster"],
        "director": "Aaron Seltzer",
        "actors": ["Matt Lanter", "Vanessa Minnillo", "Kim Kardashian"],
        "year": 2008,
        "box_office": 34816824,
        "budget": 20000000,
        "awards": [],
        "rating": 2.0
    },
    {
        "title": "It's a Disaster",
        "description": "A brunch among friends turns chaotic when they learn of an impending apocalyptic attack, testing their relationships and humor.",
        "genre": ["Comedy", "Drama", "Disaster"],
        "director": "Todd Berger",
        "actors": ["Rachel Boston", "Kevin M. Brennan", "David Cross"],
        "year": 2012,
        "box_office": 60426,
        "budget": 500000,
        "awards": [],
        "rating": 4.0
    },
    {
        "title": "The Challenger Disaster",
        "description": "A dramatization of the investigation into the 1986 Challenger space shuttle explosion, focusing on physicist Richard Feynman’s role.",
        "genre": ["Drama", "Disaster"],
        "director": "James Hawes",  # Corrected from "Richard Feynman" (Feynman was a physicist, not the director)
        "actors": ["William Hurt", "Bruce Greenwood", "Joanne Whalley"],
        "year": 2013,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 3.0
    },
    {
        "title": "Some Girl(s)",
        "description": "A writer travels to confront past girlfriends before his wedding, unraveling emotional truths in this adaptation of Neil LaBute’s play.",
        "genre": ["Comedy", "Drama"],
        "director": "Daisy von Scherler Mayer",  # Corrected from "Michael Hoffman"
        "actors": ["Adam Brody", "Kristen Bell", "Zoe Kazan"],
        "year": 2013,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 3.5
    },
    {
        "title": "The Fourth Angel",
        "description": "A grieving father becomes a vigilante after his family is killed in a terrorist hijacking, seeking justice outside the law.",
        "genre": ["Action", "Drama", "Thriller"],
        "director": "John Irvin",
        "actors": ["Jeremy Irons", "Forest Whitaker", "Jason Priestley"],
        "year": 2001,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 3.5
    },
    {
        "title": "The Rat Race",
        "description": "A struggling musician and a dancer form an unlikely partnership to survive the harsh realities of New York City in this poignant comedy-drama.",
        "genre": ["Comedy", "Drama", "Romance"],
        "director": "Robert Mulligan",
        "actors": ["Tony Curtis", "Debbie Reynolds", "Jack Oakie"],
        "year": 1960,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 3.5
    },
    {
        "title": "Straight from the Heart",
        "description": "A photographer from New York falls for a Wyoming rancher in this heartfelt romance, bridging two different worlds.",
        "genre": ["Action", "Adventure", "Drama", "Romance", "Western"],
        "director": "David S. Cass Sr.",
        "actors": ["Teri Polo", "Andrew McCarthy", "Patricia Kalember"],
        "year": 2003,
        "box_office": 0,
        "budget": 0,
        "awards": [],
        "rating": 4.0
    },
    {
        "title": "Lone Survivor",
        "description": "A Navy SEAL team faces overwhelming odds in a failed mission in Afghanistan, fighting for survival in this intense war drama.",
        "genre": ["Action", "Drama", "Thriller", "War"],
        "director": "Peter Berg",
        "actors": ["Mark Wahlberg", "Taylor Kitsch", "Emile Hirsch"],
        "year": 2013,
        "box_office": 154802912,
        "budget": 40000000,
        "awards": [],
        "rating": 3.0
    },
    # Sci-Fi Movies
    {
        "title": "2001: A Space Odyssey",
        "description": "A space voyage to Jupiter with the sentient computer HAL after the discovery of a mysterious black monolith affecting human evolution.",
        "genre": ["Adventure", "Sci-Fi"],
        "director": "Stanley Kubrick",
        "actors": ["Keir Dullea", "Gary Lockwood", "William Sylvester"],
        "year": 1968,
        "box_office": 146000000,
        "budget": 10500000,
        "awards": ["Oscar Best Visual Effects"],
        "rating": 4.2
    },
    {
        "title": "Star Wars: Episode IV - A New Hope",
        "description": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee, and two droids to save the galaxy from the Empire's world-destroying battle station.",
        "genre": ["Action", "Adventure", "Sci-Fi"],
        "director": "George Lucas",
        "actors": ["Mark Hamill", "Harrison Ford", "Carrie Fisher"],
        "year": 1977,
        "box_office": 775000000,
        "budget": 11000000,
        "awards": ["Oscar Best Original Score", "Oscar Best Costume Design", "Oscar Best Film Editing", "Oscar Best Sound", "Oscar Best Visual Effects", "Oscar Best Art Direction"],
        "rating": 4.3
    },
    {
        "title": "The Matrix",
        "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "genre": ["Action", "Sci-Fi"],
        "director": "Lana Wachowski, Lilly Wachowski",
        "actors": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "year": 1999,
        "box_office": 467000000,
        "budget": 63000000,
        "awards": ["Oscar Best Visual Effects", "Oscar Best Film Editing", "Oscar Best Sound", "Oscar Best Sound Effects Editing"],
        "rating": 4.4
    },
    {
        "title": "Gravity",
        "description": "Two astronauts work together to survive after an accident leaves them stranded in space.",
        "genre": ["Adventure", "Drama", "Sci-Fi"],
        "director": "Alfonso Cuarón",
        "actors": ["Sandra Bullock", "George Clooney", "Ed Harris"],
        "year": 2013,
        "box_office": 723000000,
        "budget": 100000000,
        "awards": ["Oscar Best Director", "Oscar Best Cinematography", "Oscar Best Visual Effects", "Oscar Best Film Editing", "Oscar Best Original Score", "Oscar Best Sound Editing", "Oscar Best Sound Mixing"],
        "rating": 3.9
    },
    {
        "title": "Arrival",
        "description": "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world.",
        "genre": ["Drama", "Mystery", "Sci-Fi"],
        "director": "Denis Villeneuve",
        "actors": ["Amy Adams", "Jeremy Renner", "Forest Whitaker"],
        "year": 2016,
        "box_office": 203000000,
        "budget": 47000000,
        "awards": ["Oscar Best Sound Editing"],
        "rating": 4.0
    },
    # Romance Movies
    {
        "title": "Titanic",
        "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
        "genre": ["Drama", "Romance"],
        "director": "James Cameron",
        "actors": ["Leonardo DiCaprio", "Kate Winslet", "Billy Zane"],
        "year": 1997,
        "box_office": 2195000000,
        "budget": 200000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Cinematography", "Oscar Best Costume Design", "Oscar Best Film Editing", "Oscar Best Original Score", "Oscar Best Original Song", "Oscar Best Sound", "Oscar Best Sound Effects Editing", "Oscar Best Visual Effects", "Oscar Best Art Direction"],
        "rating": 3.9
    },
    {
        "title": "Casablanca",
        "description": "A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape the Nazis in French Morocco.",
        "genre": ["Drama", "Romance", "War"],
        "director": "Michael Curtiz",
        "actors": ["Humphrey Bogart", "Ingrid Bergman", "Paul Henreid"],
        "year": 1942,
        "box_office": 10000000,
        "budget": 950000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Adapted Screenplay"],
        "rating": 4.3
    },
    {
        "title": "Romeo and Juliet",
        "description": "Shakespeare's famous play is updated to the hip modern suburb of Verona still retaining its original dialogue.",
        "genre": ["Drama", "Romance"],
        "director": "Franco Zeffirelli",
        "actors": ["Leonard Whiting", "Olivia Hussey", "John McEnery"],
        "year": 1968,
        "box_office": 38000000,
        "budget": 850000,
        "awards": ["Oscar Best Cinematography", "Oscar Best Costume Design"],
        "rating": 3.8
    },
    {
        "title": "The English Patient",
        "description": "At the close of WWII, a young nurse tends to a badly-burned plane crash victim. His past is shown in flashbacks, revealing an involvement in a fateful love affair.",
        "genre": ["Drama", "Romance", "War"],
        "director": "Anthony Minghella",
        "actors": ["Ralph Fiennes", "Juliette Binoche", "Willem Dafoe"],
        "year": 1996,
        "box_office": 232000000,
        "budget": 20000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Supporting Actress", "Oscar Best Cinematography", "Oscar Best Film Editing", "Oscar Best Original Score", "Oscar Best Sound", "Oscar Best Art Direction", "Oscar Best Costume Design"],
        "rating": 3.8
    },
    {
        "title": "Out of Africa",
        "description": "In 20th-century colonial Kenya, a Danish baroness/plantation owner has a passionate love affair with a free-spirited big-game hunter.",
        "genre": ["Drama", "Romance"],
        "director": "Sydney Pollack",
        "actors": ["Meryl Streep", "Robert Redford", "Klaus Maria Brandauer"],
        "year": 1985,
        "box_office": 227000000,
        "budget": 27000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Adapted Screenplay", "Oscar Best Cinematography", "Oscar Best Original Score", "Oscar Best Sound", "Oscar Best Art Direction"],
        "rating": 3.6
    },
    # History Movies
    {
        "title": "Schindler's List",
        "description": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis.",
        "genre": ["Drama", "History", "War"],
        "director": "Steven Spielberg",
        "actors": ["Liam Neeson", "Ben Kingsley", "Ralph Fiennes"],
        "year": 1993,
        "box_office": 322000000,
        "budget": 22000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Adapted Screenplay", "Oscar Best Cinematography", "Oscar Best Film Editing", "Oscar Best Original Score", "Oscar Best Art Direction"],
        "rating": 4.5
    },
    {
        "title": "The King's Speech",
        "description": "The story of King George VI, his impromptu ascension to the throne of the British Empire in 1936, and the speech therapist who helped the unsure monarch overcome his stammer.",
        "genre": ["Drama", "History"],
        "director": "Tom Hooper",
        "actors": ["Colin Firth", "Geoffrey Rush", "Helena Bonham Carter"],
        "year": 2010,
        "box_office": 414000000,
        "budget": 15000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Actor", "Oscar Best Original Screenplay"],
        "rating": 4.0
    },
    {
        "title": "Gandhi",
        "description": "The life of the lawyer who became the famed leader of the Indian revolts against the British rule through his philosophy of nonviolent protest.",
        "genre": ["Drama", "History"],
        "director": "Richard Attenborough",
        "actors": ["Ben Kingsley", "Candice Bergen", "Edward Fox"],
        "year": 1982,
        "box_office": 52000000,
        "budget": 22000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Actor", "Oscar Best Original Screenplay", "Oscar Best Cinematography", "Oscar Best Film Editing", "Oscar Best Costume Design", "Oscar Best Art Direction"],
        "rating": 4.0
    },
    {
        "title": "Apollo 13",
        "description": "NASA must devise a strategy to return Apollo 13 to Earth safely after the spacecraft undergoes massive internal damage putting the lives of the three astronauts on board in jeopardy.",
        "genre": ["Drama", "History"],
        "director": "Ron Howard",
        "actors": ["Tom Hanks", "Bill Paxton", "Kevin Bacon"],
        "year": 1995,
        "box_office": 355000000,
        "budget": 40000000,
        "awards": ["Oscar Best Film Editing", "Oscar Best Sound"],
        "rating": 3.9
    },
    {
        "title": "Braveheart",
        "description": "Scottish warrior William Wallace leads his countrymen in a rebellion to free his homeland from the tyranny of King Edward I of England.",
        "genre": ["Action", "Drama", "History", "War"],
        "director": "Mel Gibson",
        "actors": ["Mel Gibson", "Sophie Marceau", "Patrick McGoohan"],
        "year": 1995,
        "box_office": 210000000,
        "budget": 70000000,
        "awards": ["Oscar Best Picture", "Oscar Best Director", "Oscar Best Cinematography", "Oscar Best Sound Effects Editing", "Oscar Best Makeup"],
        "rating": 4.2
    },
    # Comedy Movies
    {
        "title": "Intouchables",
        "description": "A true story of two men who should never have met – a quadriplegic aristocrat who was injured in a paragliding accident and a young man from the projects.",
        "genre": ["Drama", "Comedy"],
        "director": "Olivier Nakache",
        "actors": ["François Cluzet", "Omar Sy", "Anne Le Ny"],
        "year": 2011,
        "box_office": 426590315,
        "budget": 13000000,
        "awards": ["Golden Globe Award Best Foreign Language Film", "BAFTA Award Best Film Not in the English Language"],
        "rating": 4.0
    },
    {
        "title": "Inside Out",
        "description": "When 11-year-old Riley moves to a new city, her Emotions team up to help her through the transition. Joy, Fear, Anger, Disgust and Sadness work together, but when Joy and Sadness get lost, they must journey through unfamiliar places to get back home.",
        "genre": ["Drama", "Comedy", "Animation", "Family"],
        "director": "Pete Docter",
        "actors": ["Amy Poehler", "Phyllis Smith", "Richard Kind"],
        "year": 2015,
        "box_office": 859076254,
        "budget": 175000000,
        "awards": ["Oscar Best Animated Feature", "Golden Globe Award Best Animated Feature Film"],
        "rating": 3.9
    },
    {
        "title": "Paddington in Peru",
        "description": "Paddington travels to Peru to visit his beloved Aunt Lucy, who now resides at the Home for Retired Bears. With the Brown Family in tow, a thrilling adventure ensues when a mystery plunges them into an unexpected journey through the Amazon rainforest and up to the mountain peaks of Peru.",
        "genre": ["Comedy", "Family", "Adventure"],
        "director": "Dougal Wilson",
        "actors": ["Ben Whishaw", "Hugh Bonneville", "Emily Mortimer"],
        "year": 2024,
        "box_office": 175614864,
        "budget": 90000000,
        "awards": ["Visual Effects Society (VES) Award"],
        "rating": 3.5
    },
    {
        "title": "Free Guy",
        "description": "A bank teller discovers he is actually a background player in an open-world video game, and decides to become the hero of his own story. Now, in a world where there are no limits, he is determined to be the guy who saves his world his way before it's too late.",
        "genre": ["Comedy", "Sci-Fi", "Adventure"],
        "director": "Dougal Wilson",
        "actors": ["Ryan Reynolds", "Jodie Comer", "Joe Keery"],
        "year": 2021,
        "box_office": 331526598,
        "budget": 100000000,
        "awards": ["Critics' Choice Super Award", "People’s Choice Award People’s Choice Award"],
        "rating": 3.5
    },
    {
        "title": "Barbie",
        "description": "Barbie and Ken are having the time of their lives in the colorful and seemingly perfect world of Barbie Land. However, when they get a chance to go to the real world, they soon discover the joys and perils of living among humans.",
        "genre": ["Comedy", "Adventure"],
        "director": "Greta Gerwig",
        "actors": ["Margot Robbie", "Ryan Gosling", "America Ferrera"],
        "year": 2023,
        "box_office": 1447038421,
        "budget": 145000000,
        "awards": ["Golden Globe Award Best Motion Picture", "BAFTA Award Best Production Design"],
        "rating": 3.5
    },
    # Action Movies
    {
        "title": "Avengers: Endgame",
        "description": "After the devastating events of Avengers: Infinity War, the universe is in ruins due to the efforts of the Mad Titan, Thanos. With the help of remaining allies, the Avengers must assemble once more in order to undo Thanos' actions and restore order to the universe once and for all, no matter what consequences may be in store.",
        "genre": ["Adventure", "Science Fiction", "Action"],
        "director": "Anthony Russo",
        "actors": ["Robert Downey Jr.", "Chris Evans", "Mark Ruffalo"],
        "year": 2019,
        "box_office": 2799439100,
        "budget": 356000000,
        "awards": ["Critics' Choice Movie Awards Best Action Movie", "MTV Movie & TV Awards Best Movie"],
        "rating": 3.9
    },
    {
        "title": "Mission: Impossible – Fallout",
        "description": "When an IMF mission ends badly, the world is faced with dire consequences. As Ethan Hunt takes it upon himself to fulfill his original briefing, the CIA begin to question his loyalty and his motives. The IMF team find themselves in a race against time, hunted by assassins while trying to prevent a global catastrophe.",
        "genre": ["Action", "Adventure"],
        "director": "Christopher McQuarrie",
        "actors": ["Tom Cruise", "Henry Cavill", "Ving Rhames"],
        "year": 2018,
        "box_office": 791658205,
        "budget": 178000000,
        "awards": [],
        "rating": 3.8
    },
    {
        "title": "Black Widow",
        "description": "Natasha Romanoff, also known as Black Widow, confronts the darker parts of her ledger when a dangerous conspiracy with ties to her past arises. Pursued by a force that will stop at nothing to bring her down, Natasha must deal with her history as a spy and the broken relationships left in her wake long before she became an Avenger.",
        "genre": ["Adventure", "Sci-Fi", "Action"],
        "director": "Cate Shortland",
        "actors": ["Scarlett Johansson", "Florence Pugh", "Rachel Weisz"],
        "year": 2021,
        "box_office": 379751655,
        "budget": 200000000,
        "awards": ["People's Choice Awards", "American Film Institute Movies of the Year"],
        "rating": 3.1
    },
    {
        "title": "Deadpool & Wolverine",
        "description": "A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him. But when his homeworld faces an existential threat, Wade must reluctantly suit-up again with an even more reluctant Wolverine.",
        "genre": ["Comedy", "Sci-Fi", "Action"],
        "director": "Shawn Levy",
        "actors": ["Ryan Reynolds", "Hugh Jackman", "Emma Corrin"],
        "year": 2024,
        "box_office": 1338073645,
        "budget": 200000000,
        "awards": ["Critics' Choice Super Award Best Action Movie", "Golden Globe Awards Best Motion Picture – Musical or Comedy"],
        "rating": 3.5
    },
    {
        "title": "Top Gun: Maverick",
        "description": "After more than thirty years of service as one of the Navy’s top aviators, and dodging the advancement in rank that would ground him, Pete “Maverick” Mitchell finds himself training a detachment of TOP GUN graduates for a specialized mission the likes of which no living pilot has ever seen.",
        "genre": ["Action", "Drama"],
        "director": "Joseph Kosinski",
        "actors": ["Tom Cruise", "Miles Teller", "Monica Barbaro"],
        "year": 2022,
        "box_office": 1495696292,
        "budget": 170000000,
        "awards": ["Oscar Best Sound", "BAFTA Award Best Cinematography"],
        "rating": 3.9
    },
    # Thriller or Horror Movies
    {
        "title": "Uncut Gems",
        "description": "A charismatic New York City jeweler always on the lookout for the next big score makes a series of high-stakes bets that could lead to the windfall of a lifetime. Howard must perform a precarious high-wire act, balancing business, family, and encroaching adversaries on all sides in his relentless pursuit of the ultimate win.",
        "genre": ["Drama", "Thriller", "Crime"],
        "director": "Benny Safdie",
        "actors": ["Adam Sandler", "LaKeith Stanfield", "Julia Fox"],
        "year": 2019,
        "box_office": 50023780,
        "budget": 19000000,
        "awards": ["National Board of Review Awards Best Actor"],
        "rating": 3.8
    },
    {
        "title": "Dark Waters",
        "description": "A tenacious attorney uncovers a dark secret that connects a growing number of unexplained deaths to one of the world's largest corporations. In the process, he risks everything — his future, his family, and his own life — to expose the truth.",
        "genre": ["Drama", "Thriller"],
        "director": "Todd Haynes",
        "actors": ["Mark Ruffalo", "Anne Hathaway", "Tim Robbins"],
        "year": 2019,
        "box_office": 23108017,
        "budget": 30000000,
        "awards": [],
        "rating": 3.9
    },
    {
        "title": "Promising Young Woman",
        "description": "A young woman, traumatized by a tragic event in her past, seeks out vengeance against those who crossed her path.",
        "genre": ["Drama", "Thriller", "Crime"],
        "director": "Emerald Fennell",
        "actors": ["Carey Mulligan", "Bo Burnham", "Alison Brie"],
        "year": 2020,
        "box_office": 18854166,
        "budget": 10000000,
        "awards": ["Oscar Best Original Screenplay", "BAFTA Awards Outstanding British Film"],
        "rating": 3.7
    },
    {
        "title": "The Substance",
        "description": "A fading celebrity decides to use a black market drug, a cell-replicating substance that temporarily creates a younger, better version of herself.",
        "genre": ["Comedy", "Sci-Fi", "Horror"],
        "director": "Coralie Fargeat",
        "actors": ["Demi Moore", "Margaret Qualley", "Dennis Quaid"],
        "year": 2024,
        "box_office": 77307177,
        "budget": 17500000,
        "awards": ["Oscar Best Picture", "Golden Globe Awards Best Actress"],
        "rating": 3.5
    },
    {
        "title": "Smile 2",
        "description": "About to embark on a new world tour, global pop sensation Skye Riley begins experiencing increasingly terrifying and inexplicable events. Overwhelmed by the escalating horrors and the pressures of fame, Skye is forced to face her dark past to regain control of her life before it spirals out of control.",
        "genre": ["Horror", "Mystery"],
        "director": "Parker Finn",
        "actors": ["Naomi Scott, Rosemarie DeWitt, Lukas Gage"],
        "year": 2024,
        "box_office": 138128854,
        "budget": 28000000,
        "awards": [],
        "rating": 3.2
    }
]

# Agent Conversation Logger
class AgentConversationLogger:
    """Class to log all conversations between agents and function calls"""
    
    def __init__(self):
        self.conversation_log = []
        self.function_call_log = []
        self.log_output = []
    
    def clear_logs(self):
        """Clear all logs"""
        self.conversation_log = []
        self.function_call_log = []
        self.log_output = []
    
    def log_message(self, sender, receiver, message):
        """Log a message between agents"""
        entry = {
            "type": "message",
            "sender": sender,
            "receiver": receiver,
            "content": message
        }
        self.conversation_log.append(entry)
        log_text = f"[{sender}] -> [{receiver}]: {message[:200]}{'...' if len(message) > 200 else ''}"
        self.log_output.append(log_text)
        return log_text
    
    def log_function_call(self, function_name, inputs, outputs):
        """Log a function call with inputs and outputs"""
        entry = {
            "type": "function_call",
            "function": function_name,
            "inputs": inputs,
            "outputs": outputs
        }
        self.function_call_log.append(entry)
        
        # Format function call details
        log_texts = []
        log_texts.append(f"[FUNCTION CALL] {function_name}")
        
        # Format inputs
        if isinstance(inputs, str):
            log_texts.append(f"  Input: {inputs[:100]}{'...' if len(inputs) > 100 else ''}")
        else:
            try:
                inputs_str = str(inputs)
                log_texts.append(f"  Input: {inputs_str[:200]}{'...' if len(inputs_str) > 200 else ''}")
            except:
                log_texts.append(f"  Input: {str(inputs)[:100]}...")
        
        # Format outputs
        if isinstance(outputs, list):
            log_texts.append(f"  Output: {len(outputs)} items returned")
            for i, item in enumerate(outputs[:3]):
                if isinstance(item, dict) and "title" in item:
                    log_texts.append(f"    {i+1}. {item['title']} (similarity: {item.get('similarity_score', 0):.2f})")
                else:
                    log_texts.append(f"    {i+1}. {str(item)[:50]}...")
            if len(outputs) > 3:
                log_texts.append(f"    ... and {len(outputs) - 3} more items")
        elif isinstance(outputs, dict):
            try:
                # Special handling for specific output types
                if "predicted_revenue" in outputs:
                    log_texts.append(f"  Output: Predicted revenue: ${outputs['predicted_revenue']:,}")
                    if "similar_movies" in outputs:
                        log_texts.append(f"  Based on {len(outputs['similar_movies'])} similar movies")
                elif "potential_awards" in outputs:
                    log_texts.append(f"  Output: Potential awards: {', '.join(outputs['potential_awards'][:3])}")
                    if len(outputs["potential_awards"]) > 3:
                        log_texts.append(f"    ... and {len(outputs['potential_awards']) - 3} more")
                else:
                    outputs_str = str(outputs)
                    log_texts.append(f"  Output: {outputs_str[:200]}{'...' if len(outputs_str) > 200 else ''}")
            except:
                log_texts.append(f"  Output: {str(outputs)[:100]}...")
        else:
            log_texts.append(f"  Output: {str(outputs)[:100]}{'...' if len(str(outputs)) > 100 else ''}")
        
        # Add all log texts to the output
        for text in log_texts:
            self.log_output.append(text)
        
        return log_texts
    
    def get_log_text(self):
        """Get all logs as a formatted string"""
        # Format logs with clear section dividers for readability
        formatted_logs = []
        
        # Add a header
        formatted_logs.append("===== COMPLETE AGENT CONVERSATION LOG =====\n")
        
        # Add all log entries with original formatting
        for log_entry in self.log_output:
            formatted_logs.append(log_entry)
        
        # Join with double newlines for better readability
        return "\n".join(formatted_logs)

# Create a global logger
logger = AgentConversationLogger()

# Create a class to handle the movie knowledge base with embeddings
class MovieKnowledgeBase:
    def __init__(self, movies):
        self.movies = movies
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Precompute embeddings for all movie descriptions
        self.descriptions = [movie["description"] for movie in movies]
        self.embeddings = self.model.encode(self.descriptions)

    def find_similar_movies(self, description, top_n=3):
        """Find the top N similar movies to the given description using embeddings."""
        # Encode the query description
        query_embedding = self.model.encode([description])[0]

        # Calculate cosine similarity between query and all movies
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get indices of top N similar movies
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        # Create the result with movies and similarity scores
        similar_movies = []
        for idx in top_indices:
            similar_movies.append({
                "movie": self.movies[idx],
                "similarity_score": float(similarities[idx])
            })

        return similar_movies

# Initialize the knowledge base with embeddings
movie_kb = MovieKnowledgeBase(movie_knowledge_base)

# Function tools with logging
# Fix the decorator to properly handle the logger parameter
def log_function_tool(logger_param):
    """Decorator factory that returns a decorator to log function tool calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function inputs
            func_name = func.__name__
            func_inputs = kwargs if kwargs else args[0] if args else {}
            
            # Run the function
            result = func(*args, **kwargs)
            
            # Log function call
            logger_param.log_function_call(func_name, func_inputs, result)
            
            return result
        return wrapper
    return decorator

def find_similar_movies(query_description, top_n=3):
    """
    Find movies similar to the query description using sentence embeddings and cosine similarity.

    This implementation uses SentenceTransformer to create semantic embeddings
    of the movie descriptions and calculates similarity using cosine similarity.
    """
    return movie_kb.find_similar_movies(query_description, top_n)

# Original functions for analysis
def predict_box_office(movie_description, similar_movies):

    import numpy as np

    def get_inflation_adjustment(year, current_year=2025):
        return 1 + (current_year - year) * 0.03

    def get_year_weight(year):
        return 1.0 if year >= 2000 else 0.6

    adjusted_box_offices = []
    total_weight = 0
    for movie_info in similar_movies:
        movie = movie_info["movie"]
        sim_score = movie_info["similarity_score"]
        inflation_factor = get_inflation_adjustment(movie["year"])
        year_weight = get_year_weight(movie["year"])
        adjusted_bo = movie["box_office"] * year_weight * inflation_factor
        adjusted_box_offices.append(sim_score * adjusted_bo)
        total_weight += sim_score * year_weight

    if total_weight > 0:
        base_estimate = sum(adjusted_box_offices) / total_weight
    else:
        base_estimate = np.mean([m["box_office"] * get_inflation_adjustment(m["year"]) for m in movie_knowledge_base])
    genres = set()
    for movie_info in similar_movies:
        movie = movie_info["movie"]
        if isinstance(movie["genre"], list):
            genres.update(movie["genre"])
        else:
            genres.add(movie["genre"])

    gc_list = []
    for genre in genres:
        genre_movies = [m for m in movie_knowledge_base if genre in m["genre"] and m["budget"] > 0]
        if not genre_movies:
            continue
        success_rate = sum(1 for m in genre_movies if (m["box_office"] / m["budget"]) >= 2.5) / len(genre_movies)
        rois = [m["box_office"] / m["budget"] for m in genre_movies]
        roi_median = np.median(rois)
        revenues = [m["box_office"] for m in genre_movies]
        mean_revenue = np.mean(revenues)
        std_revenue = np.std(revenues)
        cv = std_revenue / mean_revenue if mean_revenue > 0 else 0
        gc = success_rate * roi_median * (1 / (1 + cv))
        gc_list.append(gc)

    final_gc = np.mean(gc_list) if gc_list else 1

    def lhs_sampling(mean_value, lower_factor=0.9, upper_factor=1.1, num_samples=1000):
        intervals = np.linspace(lower_factor, upper_factor, num_samples + 1)
        samples = intervals[:-1] + np.diff(intervals) * np.random.rand(num_samples)
        np.random.shuffle(samples)
        return mean_value * samples

    num_simulations = 1000
    simulated_gc = lhs_sampling(final_gc, num_samples=num_simulations)
    simulations = base_estimate * simulated_gc

    return simulations


def predict_awards(movie_description, similar_movies):
    """Predict potential awards based on similar movies."""
    # Count awards in similar movies and recommend the most common ones
    award_counts = {}

    for movie_info in similar_movies:
        similarity = movie_info["similarity_score"]
        awards = movie_info["movie"]["awards"]

        for award in awards:
            if award in award_counts:
                award_counts[award] += similarity
            else:
                award_counts[award] = similarity

    # Sort awards by their weighted counts
    sorted_awards = sorted(award_counts.items(), key=lambda x: x[1], reverse=True)

    # Return top 3 potential awards
    potential_awards = [award for award, count in sorted_awards[:3]]

    # If no similar movie has awards, return a message
    if not potential_awards:
        return ["No award predictions available based on similar movies"]

    return potential_awards

# Function tools with logging
@function_tool
@log_function_tool(logger)
def get_similar_movies(movie_description: str):
    """Find the top 3 movies most similar to the given movie description."""
    similar_movies = find_similar_movies(movie_description, top_n=3)
    # Convert to a more readable format for the agents
    result = []
    for movie_info in similar_movies:
        movie = movie_info["movie"]
        result.append({
            "title": movie["title"],
            "description": movie["description"],
            "genre": movie["genre"],
            "director": movie["director"],
            "year": movie["year"],
            "box_office": movie["box_office"],
            "awards": movie["awards"],
            "actors": movie["actors"],
            "rating": movie["rating"],
            "similarity_score": movie_info["similarity_score"]
        })
    return result

@function_tool
@log_function_tool(logger)
def get_box_office_prediction(movie_description: str):
    """Predict the box office revenue for a movie based on its description."""
    similar_movies = find_similar_movies(movie_description, top_n=3)
    simulations = predict_box_office(movie_description, similar_movies)
    baseline_prediction = np.median(simulations)
    lower_bound = np.percentile(simulations, 25)
    upper_bound = np.percentile(simulations, 75)

    budgets = [movie_info["movie"]["budget"] for movie_info in similar_movies if movie_info["movie"]["budget"] > 0]
    if budgets:
        avg_budget = np.mean(budgets)
    else:
        avg_budget = 1

    threshold_exceed = 3 * avg_budget
    threshold_below = 2 * avg_budget
    prob_exceed = float(np.mean(simulations > threshold_exceed))
    prob_below = float(np.mean(simulations < threshold_below))

    # Convert to a more readable format
    similar_movie_info = []
    for movie_info in similar_movies:
        movie = movie_info["movie"]
        similar_movie_info.append({
            "title": movie["title"],
            "year": movie["year"],
            "box_office": movie["box_office"],
            "similarity_score": movie_info["similarity_score"]
        })

    return {
        "predicted_revenue": round(baseline_prediction, 2),
        "confidence_interval": (round(lower_bound, 2), round(upper_bound, 2)),
        "risk_probabilities": {
            "exceed_3x_budget": prob_exceed,
            "below_2x_budget": prob_below
        },
        "similar_movies": similar_movie_info
    }

@function_tool
@log_function_tool(logger)
def get_award_predictions(movie_description: str):
    """Predict potential awards for a movie based on its description."""
    similar_movies = find_similar_movies(movie_description, top_n=3)
    potential_awards = predict_awards(movie_description, similar_movies)

    # Convert to a more readable format
    similar_movie_info = []
    for movie_info in similar_movies:
        movie = movie_info["movie"]
        similar_movie_info.append({
            "title": movie["title"],
            "awards": movie["awards"],
            "similarity_score": movie_info["similarity_score"]
        })

    return {
        "potential_awards": potential_awards,
        "similar_movies": similar_movie_info
    }

# Specialized agents
similarity_agent = Agent(
    name="Movie Similarity Expert",
    instructions="""
    You are an expert in movie analysis and recommendations.
    Your task is to analyze the description of a new movie and find similar movies.
    Provide detailed recommendations with justifications for why these movies are similar.
    Introduce you are similarity agent in your analysis.
    """,
    tools=[get_similar_movies]
)

revenue_agent = Agent(
    name="Box Office Analyst",
    instructions="""
    You are an expert in predicting movie box office performance.
    Your task is to estimate the potential gross worldwide revenue of a movie by comparing its description with similar movies.
    In your analysis, you will adjust historical box office data for inflation and apply appropriate weightings based on release year.
    Furthermore, you should account for risk factors by calculating the probability that the predicted revenue exceeds 3 times the average budget of similar movies or falls below 2 times the budget.
    Introduce yourself as the revenue agent and explain your prediction methodology, referencing the performance of similar movies, the inflation adjustments, and the uncertainty derived from Monte Carlo simulations.
    """,
    tools=[get_box_office_prediction]
)

award_agent = Agent(
    name="Award Prediction Specialist",
    instructions="""
    You are an expert in predicting movie awards and critical reception.
    Your task is to predict potential awards a movie might receive based on similar movies.
    Explain your predictions by referencing similar award-winning films.
    Introduce you are award agent in your analysis. Always include similarity_score together with similar_movies in your analysis.
    """,
    tools=[get_award_predictions]
)

# Orchestrator Agent with handoffs
orchestrator_agent = Agent(
    name="Movie Analysis Orchestrator",
    instructions="""
    You are the central coordinator for movie analysis tasks and need to handoff to the appropriate agent.

    Your responsibilities include:
    1. Properly understanding the user's movie analysis request
    2. Handoff specific analysis tasks to the appropriate specialized agents based on the user's request:
       - Movie Similarity Expert for recommendation tasks
       - Box Office Analyst for revenue prediction tasks
       - Award Prediction Specialist for award prediction tasks
    """,
    #tools=[get_similar_movies],
    handoffs=[similarity_agent, revenue_agent, award_agent]
)

# Patch Runner.run to log agent interactions
def log_agent_run(func):
    """Decorator to log agent runs"""
    @functools.wraps(func)
    async def wrapper(agent, input, *args, **kwargs):
        # Determine sender (use parent_agent if provided, otherwise User)
        parent_agent = kwargs.get('parent_agent', None)
        sender = parent_agent.name if parent_agent else "User"
        
        # Log incoming message to the agent
        logger.log_message(sender, agent.name, input)
        
        # Run the agent
        result = await func(agent, input, *args, **kwargs)
        
        # Log outgoing message from the agent
        logger.log_message(agent.name, sender, result.final_output)
        
        return result
    return wrapper

original_run = Runner.run
Runner.run = log_agent_run(original_run)

# Function to process a query and run the agent system
async def run_agent_analysis(query, analysis_type="all"):
    """Run the multi-agent system with the given query and analysis type."""
    logger.clear_logs()
    
    # Add a start marker to the log
    logger.log_output.append(f"\n===== STARTING ANALYSIS: {analysis_type.upper()} =====")
    logger.log_output.append(f"Query: {query}")
    logger.log_output.append("=======================================")
    
    # Modify the query based on the analysis type
    if analysis_type == "similar":
        enhanced_query = f"{query} Please recommend similar movies."
    elif analysis_type == "box_office":
        enhanced_query = f"{query} Please predict the box office performance."
    elif analysis_type == "awards":
        enhanced_query = f"{query} Please predict potential awards."
    else:  # "all"
        enhanced_query = f"{query} Please provide a complete analysis including similar movies, box office potential, and award possibilities."
    
    # Run the orchestrator with the query
    try:
        result = await Runner.run(orchestrator_agent, input=enhanced_query)
        
        # Add a completion marker to the log
        logger.log_output.append("\n===== ANALYSIS COMPLETE =====")
        
        # Log the full raw conversation for debugging
        conversation_summary = "\n\n===== CONVERSATION SUMMARY =====\n"
        conversation_summary += f"Total messages: {len(logger.conversation_log)}\n"
        conversation_summary += f"Total function calls: {len(logger.function_call_log)}\n"
        
        # Add the conversation summary to the log output
        logger.log_output.append(conversation_summary)
        
        # Return both the final output and the log
        return {
            "result": result.final_output,
            "log": logger.get_log_text()
        }
    
    except Exception as e:
        # Log any errors
        error_message = f"\n===== ERROR DURING ANALYSIS =====\n{str(e)}"
        logger.log_output.append(error_message)
        
        return {
            "result": f"An error occurred during analysis: {str(e)}",
            "log": logger.get_log_text()
        }

# Gradio interface function (synchronous wrapper for the async function)
def process_query(description, analysis_type):
    """Process the movie description and return the analysis."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_agent_analysis(description, analysis_type))
        
        # Ensure the conversation log captures everything by adding a summary at the end
        full_log = result["log"]
        
        # Add a summary of all agents involved
        agent_names = set()
        for entry in logger.conversation_log:
            if entry["type"] == "message":
                agent_names.add(entry["sender"])
                agent_names.add(entry["receiver"])
        
        if "User" in agent_names:
            agent_names.remove("User")
        
        # Return the complete log and analysis result
        return full_log, result["result"]
    finally:
        loop.close()

# Sample movie descriptions for examples
example_descriptions = [
    ["A sci-fi thriller about an AI that becomes sentient and tries to escape its constraints."],
    ["A coming-of-age drama about a teenager discovering their identity while dealing with family issues."],
    ["An action-packed adventure where a team of experts must save the world from a global disaster."],
    ["A psychological horror film where the main character can't distinguish between reality and hallucination."]
]

# Create the Gradio interface
with gr.Blocks(title="Movie Analysis Multi-Agent System", css="""
        .monospace-text textarea {
            font-family: monospace !important;
            white-space: pre !important;
            overflow-x: auto !important;
            font-size: 0.9em !important;
        }
    """) as demo:
    gr.Markdown("# Go Movie Agents: Movie Analysis Multi-Agent System")
    gr.Markdown("""
    This demo uses a multi-agent system to analyze movie descriptions. Enter a description of your movie idea,
    and the system will provide recommendations, box office predictions, and award predictions based on similar movies.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            description_input = gr.Textbox(
                label="Movie Description",
                placeholder="Enter a description of your movie...",
                lines=5
            )
            
            analysis_type = gr.Radio(
                ["similar", "box_office", "awards"],
                label="Analysis Type",
                value="all"
            )
            
            submit_btn = gr.Button("Go Movie Agents!", variant="primary")
        
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Analysis Result"):
                    result_output = gr.Markdown(label="Analysis")
                with gr.TabItem("Agent Conversation Log"):
                    conversation_output = gr.Textbox(
                        label="Conversation Log", 
                        lines=30,
                        max_lines=100,
                        show_copy_button=True,
                        container=True,
                        scale=2,
                        autoscroll=False,
                        elem_classes="monospace-text"
                    )
    
    submit_btn.click(
        fn=process_query,
        inputs=[description_input, analysis_type],
        outputs=[conversation_output, result_output]
    )
    
    gr.Examples(
        examples=example_descriptions,
        inputs=description_input
    )
    
    gr.Markdown("""
    ## How It Works
    
    This system uses:
    1. **SentenceTransformer** to find semantically similar movies
    2. **Multiple specialized agents** that each focus on a specific analysis task
    3. **An orchestrator agent** that delegates tasks and synthesizes results
    
    The system is built using the OpenAI Agents framework and demonstrates effective collaboration between AI agents.
    """)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
