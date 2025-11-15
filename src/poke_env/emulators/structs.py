
# variables are species, type1, type2, level, current_hp, status, evs (dict), ivs (dict), stats (dict), moves (list)
class Pokemon:
    def __init__(self, species, type1, type2, level, current_hp, status, hp_ev, attack_ev, defense_ev, speed_ev, special_ev, attack_defense_iv, speed_special_iv, max_hp, attack, defense, speed, special, move1, move2, move3, move4, move_1_pp, move_2_pp, move_3_pp, move_4_pp):
        self.species = species
        self.type1 = type1
        self.type2 = type2
        self.level = level
        self.current_hp = current_hp
        self.status = status
        self.evs = {
            'hp': hp_ev,
            'attack': attack_ev,
            'defense': defense_ev,
            'speed': speed_ev,
            'special': special_ev
        }
        self.ivs = {
            'attack_defense': attack_defense_iv,
            'speed_special': speed_special_iv
        }
        self.stats = {
            'max_hp': max_hp,
            'attack': attack,
            'defense': defense,
            'speed': speed,
            'special': special
        }
        self.moves = [
            (move1, move_1_pp),
            (move2, move_2_pp),
            (move3, move_3_pp),
            (move4, move_4_pp)
        ]
        

    def __str__(self):
        return f"Pokemon(species={self.species}, level={self.level})"
