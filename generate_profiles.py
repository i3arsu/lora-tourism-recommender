"""Generate 5000 diverse tourist profiles for LoRA training.

Tags are extracted from Pula_data.json and pula_ostalo_new_edited.json and grouped
into thematic categories so each profile can be built from a coherent subset.
"""

import json
import random

random.seed(42)

# ---- Tag taxonomy (extracted from the two datasets) ------------------------

CUISINES = [
    "mediterranean", "italian", "croatian", "istrian", "balkan", "asian",
    "chinese", "international", "regional", "local", "neapolitan_pizza",
]

FOOD_ITEMS = [
    "seafood", "fish", "meat", "steak", "grill", "barbecue", "pizza", "pasta",
    "burgers", "sandwiches", "kebab", "cevapi", "truffles", "salads",
    "desserts", "cakes", "ice_cream", "pancakes", "waffles", "fries", "brunch",
    "breakfast", "bakery", "smoothies", "wok", "daily_menu",
]

DIETARY = ["vegetarian", "vegan", "gluten_free", "healthy"]

DRINKS = ["coffee", "wine", "beer", "cocktails"]

DINING_VIBE = ["city_center", "music", "nightlife", "self_service",
               "hotel_restaurant", "outdoor_seating", "indoor_seating"]

HERITAGE = [
    "fortress", "historic", "roman", "ancient", "prehistoric",
    "archeological_site", "ruins", "temple", "military_history", "landmark",
    "monument", "memorial", "museum", "gallery", "artwork", "sculpture",
    "statue", "aviation", "industrial", "city_gate", "amphitheatre",
    "geopaleontological_site", "church",
]

NATURE = [
    "coastal", "hilltop", "panoramic_view", "viewpoint", "nature",
    "national_park", "island", "park", "garden", "beach", "sandy", "pebble",
    "rocky", "stone", "concrete", "near_beach", "seaside", "underground",
    "labyrinth", "meditative",
]

ACTIVITIES = ["events", "concerts", "kayaking", "sports", "playground",
              "interactive", "nightlife"]

FAMILY_ACCESS = ["family_friendly", "pet_friendly", "adults_only",
                 "wheelchair_accessible"]

ACCOMMODATION = [
    "hotel", "apartment", "hostel", "guest_house", "chalet", "villa",
    "resort", "camp", "pool", "wellness", "free_parking", "near_attractions",
    "garden",
]

ANIMALS = ["zoo", "safari_park", "animals", "aquarium"]

PLACE_TYPES = [
    "restaurant", "cafe", "fast_food", "pub", "fort", "camp_site",
    "archaeological_site", "zoo", "park", "attraction", "museum", "apartment",
    "hotel", "hostel", "gallery", "monument", "city_gate", "ruins", "beach",
    "artwork", "castle", "fountain", "marina", "aquarium", "playground",
    "memorial", "viewpoint", "guest_house", "chalet",
]

PRICE = ["low", "medium", "high"]

ALL_CATEGORIES = {
    "cuisines": CUISINES,
    "food_items": FOOD_ITEMS,
    "dietary": DIETARY,
    "drinks": DRINKS,
    "dining_vibe": DINING_VIBE,
    "heritage": HERITAGE,
    "nature": NATURE,
    "activities": ACTIVITIES,
    "family_access": FAMILY_ACCESS,
    "accommodation": ACCOMMODATION,
    "animals": ANIMALS,
    "place_types": PLACE_TYPES,
}

# Natural conflict pairs used to build plausible dislikes
CONFLICTS = {
    "vegan": ["meat", "steak", "barbecue", "seafood", "fish", "kebab", "cevapi"],
    "vegetarian": ["meat", "steak", "barbecue", "kebab", "cevapi"],
    "gluten_free": ["pizza", "pasta", "bakery", "sandwiches", "burgers"],
    "healthy": ["fast_food", "burgers", "fries", "desserts", "cakes"],
    "adults_only": ["family_friendly", "playground", "kids"],
    "family_friendly": ["nightlife", "pub", "adults_only", "cocktails"],
    "quiet": ["nightlife", "concerts", "events", "music", "crowdy"],
    "pet_friendly": ["closed_area"],
}

# ---- Persona archetypes ----------------------------------------------------
# Each archetype picks from a few categories with characteristic weights.
# "breadth" controls how many tags it pulls in -> produces narrow to wide
# profiles as requested.

ARCHETYPES = [
    {
        "name": "foodie_gourmet",
        "likes_from": ["cuisines", "food_items", "drinks", "dining_vibe"],
        "dislikes_bias": ["fast_food", "self_service", "fries"],
        "age_range": (28, 65),
    },
    {
        "name": "budget_backpacker",
        "likes_from": ["accommodation", "food_items", "nature", "activities"],
        "dislikes_bias": ["adults_only", "wellness", "high"],
        "age_range": (18, 32),
    },
    {
        "name": "history_buff",
        "likes_from": ["heritage", "place_types", "dining_vibe"],
        "dislikes_bias": ["nightlife", "burgers", "fast_food"],
        "age_range": (35, 75),
    },
    {
        "name": "beach_lover",
        "likes_from": ["nature", "activities", "accommodation"],
        "dislikes_bias": ["museum", "underground", "closed_area"],
        "age_range": (20, 55),
    },
    {
        "name": "family_traveler",
        "likes_from": ["family_access", "activities", "animals",
                       "accommodation", "food_items"],
        "dislikes_bias": ["nightlife", "adults_only", "cocktails", "pub"],
        "age_range": (30, 50),
    },
    {
        "name": "nightlife_partygoer",
        "likes_from": ["activities", "drinks", "dining_vibe", "food_items"],
        "dislikes_bias": ["museum", "meditative", "wellness"],
        "age_range": (19, 35),
    },
    {
        "name": "nature_hiker",
        "likes_from": ["nature", "activities", "animals"],
        "dislikes_bias": ["city_center", "nightlife", "industrial"],
        "age_range": (22, 68),
    },
    {
        "name": "wellness_seeker",
        "likes_from": ["accommodation", "nature", "dietary", "drinks"],
        "dislikes_bias": ["fast_food", "nightlife", "fries", "kebab"],
        "age_range": (30, 70),
    },
    {
        "name": "art_culture_lover",
        "likes_from": ["heritage", "dining_vibe", "drinks"],
        "dislikes_bias": ["fast_food", "safari_park"],
        "age_range": (25, 72),
    },
    {
        "name": "adventure_athlete",
        "likes_from": ["activities", "nature", "food_items"],
        "dislikes_bias": ["adults_only", "meditative", "wellness"],
        "age_range": (20, 45),
    },
    {
        "name": "vegan_traveler",
        "likes_from": ["dietary", "cuisines", "drinks", "nature"],
        "dislikes_bias": ["meat", "steak", "barbecue", "kebab", "cevapi",
                          "seafood", "fish"],
        "age_range": (20, 55),
    },
    {
        "name": "luxury_couple",
        "likes_from": ["accommodation", "cuisines", "drinks", "heritage"],
        "dislikes_bias": ["hostel", "camp", "self_service", "fast_food",
                          "crowdy"],
        "age_range": (28, 65),
    },
    {
        "name": "retired_cruiser",
        "likes_from": ["heritage", "nature", "cuisines", "accommodation"],
        "dislikes_bias": ["nightlife", "hostel", "kayaking"],
        "age_range": (58, 82),
    },
    {
        "name": "digital_nomad",
        "likes_from": ["dining_vibe", "drinks", "accommodation", "food_items"],
        "dislikes_bias": ["crowdy", "closed_area"],
        "age_range": (24, 42),
    },
    {
        "name": "pet_owner",
        "likes_from": ["family_access", "nature", "accommodation"],
        "dislikes_bias": ["closed_area", "adults_only"],
        "age_range": (25, 60),
    },
    {
        "name": "romantic_couple",
        "likes_from": ["cuisines", "drinks", "heritage", "nature",
                       "accommodation"],
        "dislikes_bias": ["hostel", "fast_food", "playground"],
        "age_range": (22, 50),
    },
    {
        "name": "niche_specialist",   # very narrow, deep interest
        "likes_from": ["heritage"],
        "dislikes_bias": ["nightlife", "fast_food"],
        "age_range": (30, 70),
    },
    {
        "name": "casual_generalist",  # very wide interests
        "likes_from": list(ALL_CATEGORIES.keys()),
        "dislikes_bias": [],
        "age_range": (18, 75),
    },
]

FIRST_NAMES_F = [
    "Ana", "Marija", "Petra", "Lena", "Sofia", "Emma", "Olivia", "Isabella",
    "Chloe", "Mia", "Laura", "Nina", "Julia", "Sara", "Elena", "Hanna",
    "Amélie", "Giulia", "Sophie", "Katarina", "Ivana", "Marta", "Lucia",
    "Yuki", "Priya", "Fatima", "Zara", "Aisha", "Nora", "Klara",
]
FIRST_NAMES_M = [
    "Marko", "Luka", "Ivan", "Mateo", "Noah", "Liam", "Daniel", "Alex",
    "Leon", "Tomás", "Mario", "Stefan", "Hugo", "Oliver", "Felix", "Pavel",
    "Jakob", "Andrea", "Kenji", "Arjun", "Omar", "Jonas", "Mikhail", "Pedro",
    "Niko", "Filip", "Damian", "Raffaele", "Lars", "Enzo",
]
NATIONALITIES = [
    "German", "Italian", "French", "Austrian", "Slovenian", "Croatian",
    "Dutch", "British", "American", "Canadian", "Australian", "Polish",
    "Czech", "Hungarian", "Swedish", "Norwegian", "Danish", "Finnish",
    "Spanish", "Portuguese", "Belgian", "Swiss", "Irish", "Greek", "Serbian",
    "Bosnian", "Japanese", "Korean", "Indian", "Brazilian", "Mexican",
    "Turkish", "Romanian",
]
TRAVEL_STYLES = [
    "solo", "couple", "family with kids", "group of friends", "backpacker",
    "with partner", "with parents", "small group", "honeymoon",
]

def pick_tags(archetype, breadth):
    """Pick a set of liked tags from the archetype's categories."""
    cat_names = archetype["likes_from"]
    # breadth: narrow(2-4), medium(5-9), wide(10-18), very_wide(18-28)
    total_target = {
        "narrow": random.randint(2, 4),
        "medium": random.randint(5, 9),
        "wide": random.randint(10, 18),
        "very_wide": random.randint(18, 28),
    }[breadth]

    likes = set()
    attempts = 0
    while len(likes) < total_target and attempts < 200:
        attempts += 1
        cat = random.choice(cat_names)
        pool = ALL_CATEGORIES[cat]
        likes.add(random.choice(pool))
    return list(likes)


def build_dislikes(likes, archetype):
    dislikes = set()
    # Conflicts implied by likes
    for tag in likes:
        if tag in CONFLICTS:
            for c in CONFLICTS[tag]:
                if c not in likes:
                    dislikes.add(c)
    # Archetype bias
    for tag in archetype["dislikes_bias"]:
        if tag not in likes and random.random() < 0.55:
            dislikes.add(tag)
    # Occasional random mild dislikes
    if random.random() < 0.45:
        extra_pool = random.choice(list(ALL_CATEGORIES.values()))
        pick = random.choice(extra_pool)
        if pick not in likes:
            dislikes.add(pick)
    # Cap
    dislikes = list(dislikes)
    random.shuffle(dislikes)
    return dislikes[: random.randint(0, 6)]


# ---- Input question generation --------------------------------------------

PRICE_ADJ = {
    "low": ["cheap", "affordable", "budget-friendly", "inexpensive"],
    "medium": ["reasonably priced", "mid-range", "decent"],
    "high": ["upscale", "luxurious", "high-end", "fancy", "premium"],
}

PERSONA_QUESTION_TEMPLATES = {
    "foodie_gourmet": [
        "Can you recommend a {price} place in Pula for great {food}?",
        "Where can I find the best {food} in town?",
        "What's a top-rated spot for {food} lovers?",
        "Any {price} restaurants with excellent {food} you'd suggest?",
    ],
    "budget_backpacker": [
        "Where can I grab a {price} bite with {food}?",
        "Any {price} places near the {nature} worth checking out?",
        "Can you suggest a cheap spot to eat {food}?",
        "What are some budget-friendly things to do around here?",
    ],
    "history_buff": [
        "What historical sites should I visit in Pula?",
        "Can you recommend places with {heritage} to explore?",
        "Where can I learn about the {heritage} of this region?",
        "Any must-see {heritage} attractions nearby?",
    ],
    "beach_lover": [
        "Which beaches around Pula are worth visiting?",
        "Can you suggest a nice {nature} spot for the day?",
        "Where's the best place to spend a sunny day by the sea?",
        "Any good {nature} areas to relax at?",
    ],
    "family_traveler": [
        "What are some family-friendly activities in Pula?",
        "Where can I take the kids for a fun day out?",
        "Any {price} family-friendly restaurants you'd recommend?",
        "What's a good spot that both kids and adults would enjoy?",
    ],
    "nightlife_partygoer": [
        "Where's the best nightlife in Pula?",
        "Any good spots for {food} and drinks late at night?",
        "Can you suggest a lively place with {food}?",
        "Where should I go out tonight for a good time?",
    ],
    "nature_hiker": [
        "Where can I go for a nice walk in nature?",
        "Any good {nature} spots for hiking around Pula?",
        "Can you recommend a scenic outdoor place to visit?",
        "What's a peaceful nature destination I should check out?",
    ],
    "wellness_seeker": [
        "Where can I find a relaxing {price} wellness spot?",
        "Can you recommend a place with {nature} and calm vibes?",
        "Any healthy food places you'd suggest?",
        "Where can I go to unwind and feel refreshed?",
    ],
    "art_culture_lover": [
        "What cultural attractions should I visit in Pula?",
        "Can you recommend a museum or gallery to explore?",
        "Where can I experience the local art and {heritage} scene?",
        "Any interesting cultural events or venues nearby?",
    ],
    "adventure_athlete": [
        "Where can I do some {activity} around Pula?",
        "Any good spots for an active, adventurous day?",
        "Can you suggest a place for outdoor sports?",
        "What's an exciting activity to try here?",
    ],
    "vegan_traveler": [
        "Where can I find great vegan food in Pula?",
        "Any plant-based restaurants you'd recommend?",
        "Can you suggest a {price} vegan-friendly spot?",
        "What's the best place for vegan or vegetarian food?",
    ],
    "luxury_couple": [
        "Can you recommend something upscale and luxurious?",
        "Where's a fine dining spot for a special evening?",
        "Any high-end {food} restaurants you'd suggest?",
        "What's the most elegant place to stay or eat in Pula?",
    ],
    "retired_cruiser": [
        "Can you suggest a nice, quiet place to have a meal?",
        "Where can we enjoy some traditional {food}?",
        "Any {heritage} sites that are easy to visit?",
        "What's a pleasant {price} restaurant for a relaxed evening?",
    ],
    "digital_nomad": [
        "Where's a good cafe to work from with great coffee?",
        "Can you recommend a {price} spot to grab {food} and work?",
        "Any cozy cafes with wifi and good {food}?",
        "What's a comfortable place to spend a few hours with a laptop?",
    ],
    "pet_owner": [
        "Are there any pet-friendly places you'd recommend?",
        "Where can I go that allows dogs?",
        "Any nice outdoor spots where I can bring my pet?",
        "Can you suggest a pet-friendly restaurant or park?",
    ],
    "romantic_couple": [
        "Where's a romantic {price} spot for dinner?",
        "Can you suggest a special place for a couple?",
        "Any charming restaurants with {food} and a nice atmosphere?",
        "Where would you recommend for a romantic evening?",
    ],
    "niche_specialist": [
        "Where can I see the best {heritage} in Pula?",
        "Any recommendations for {heritage} enthusiasts?",
        "Can you point me to a deep dive into the local {heritage}?",
        "What's the must-visit spot for someone interested in {heritage}?",
    ],
    "casual_generalist": [
        "What's something fun to do in Pula?",
        "Can you recommend something good for today?",
        "Any top things I shouldn't miss around here?",
        "What would you suggest for a nice day out?",
    ],
}


def build_input(profile, archetype_name):
    """Generate a natural question reflecting the profile's preferences."""
    likes = profile["likes"]
    price = profile["price_preference"]

    def pick_from(cat_list):
        options = [t for t in likes if t in cat_list]
        return random.choice(options) if options else None

    food = pick_from(CUISINES + FOOD_ITEMS + DIETARY)
    heritage = pick_from(HERITAGE)
    nature = pick_from(NATURE)
    activity = pick_from(ACTIVITIES + NATURE)

    templates = PERSONA_QUESTION_TEMPLATES[archetype_name]
    # Filter templates that need a slot the profile doesn't have
    usable = []
    for t in templates:
        needs_food = "{food}" in t
        needs_heritage = "{heritage}" in t
        needs_nature = "{nature}" in t
        needs_activity = "{activity}" in t
        if needs_food and not food:
            continue
        if needs_heritage and not heritage:
            continue
        if needs_nature and not nature:
            continue
        if needs_activity and not activity:
            continue
        usable.append(t)
    if not usable:
        usable = [t for t in templates if "{" not in t or "{price}" in t]
    if not usable:
        usable = templates

    template = random.choice(usable)
    price_word = random.choice(PRICE_ADJ[price])
    return template.format(
        price=price_word,
        food=food or "good food",
        heritage=heritage or "history",
        nature=nature or "nature",
        activity=activity or "activities",
    )


def build_profile(pid):
    archetype = random.choice(ARCHETYPES)
    # Weighted breadth: skew toward medium/wide but keep narrow & very_wide.
    breadth = random.choices(
        ["narrow", "medium", "wide", "very_wide"],
        weights=[20, 45, 25, 10],
    )[0]
    if archetype["name"] == "niche_specialist":
        breadth = "narrow"
    if archetype["name"] == "casual_generalist":
        breadth = random.choice(["wide", "very_wide"])

    gender = random.choices(["female", "male"], weights=[50, 50])[0]
    if gender == "female":
        name = random.choice(FIRST_NAMES_F)
    else:
        name = random.choice(FIRST_NAMES_M)

    age_lo, age_hi = archetype["age_range"]
    age = random.randint(age_lo, age_hi)

    likes = pick_tags(archetype, breadth)
    dislikes = build_dislikes(likes, archetype)

    # Price preference correlated with persona
    if archetype["name"] in ("luxury_couple",):
        price_pref = random.choices(PRICE, weights=[1, 3, 8])[0]
    elif archetype["name"] in ("budget_backpacker", "digital_nomad"):
        price_pref = random.choices(PRICE, weights=[7, 3, 1])[0]
    else:
        price_pref = random.choices(PRICE, weights=[3, 6, 3])[0]

    # Min rating they care about
    min_rating = random.choice([3.5, 3.8, 4.0, 4.2, 4.5])

    travel_style = random.choice(TRAVEL_STYLES)
    nationality = random.choice(NATIONALITIES)

    profile = {
        "id": pid,
        "name": name,
        "gender": gender,
        "age": age,
        "nationality": nationality,
        "travel_style": travel_style,
        "persona": archetype["name"],
        "likes": sorted(likes),
        "dislikes": sorted(dislikes),
        "price_preference": price_pref,
        "min_rating": min_rating,
    }
    profile["input"] = build_input(profile, archetype["name"])
    return profile


def main():
    profiles = [build_profile(i + 1) for i in range(5000)]
    with open("user_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    # Quick distribution sanity check
    from collections import Counter
    personas = Counter(p["persona"] for p in profiles)
    genders = Counter(p["gender"] for p in profiles)
    print("profiles:", len(profiles))
    print("personas:", personas)
    print("genders:", genders)
    print("avg likes:",
          sum(len(p["likes"]) for p in profiles) / len(profiles))
    print("avg dislikes:",
          sum(len(p["dislikes"]) for p in profiles) / len(profiles))


if __name__ == "__main__":
    main()
