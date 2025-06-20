import torch
from .BEATs import BEATs, BEATsConfig
from pathlib import Path


class BeatsWrapper(torch.nn.Module):
    def __init__(
        self,
        checkpoint=Path(__file__).parent / "BEATs_iter1_finetuned_on_AS2M_cpt1.pt",
    ):
        super().__init__()
        # load the fine-tuned checkpoints
        checkpoint = torch.load(checkpoint)
        cfg = BEATsConfig(checkpoint["cfg"])
        mdl_impl = BEATs(cfg)
        mdl_impl.load_state_dict(checkpoint["model"])
        mdl_impl.eval()
        self.mappings = [
            134,
            84,
            137,
            474,
            506,
            141,
            140,
            143,
            138,
            462,
            139,
            459,
            142,
            519,
            509,
            194,
            192,
            189,
            9,
            10,
            0,
            378,
            413,
            467,
            151,
            171,
            164,
            243,
            162,
            239,
            172,
            165,
            168,
            161,
            14,
            433,
            301,
            304,
            360,
            169,
            92,
            166,
            120,
            119,
            332,
            330,
            328,
            329,
            505,
            27,
            11,
            507,
            445,
            368,
            489,
            498,
            346,
            453,
            512,
            95,
            86,
            72,
            26,
            281,
            32,
            247,
            481,
            218,
            354,
            357,
            230,
            376,
            398,
            510,
            426,
            435,
            434,
            401,
            358,
            427,
            430,
            24,
            300,
            317,
            298,
            446,
            64,
            522,
            299,
            349,
            62,
            321,
            308,
            337,
            68,
            222,
            431,
            484,
            326,
            526,
            215,
            159,
            163,
            371,
            370,
            288,
            386,
            525,
            220,
            223,
            392,
            469,
            277,
            353,
            312,
            313,
            315,
            265,
            444,
            410,
            404,
            170,
            58,
            508,
            292,
            381,
            339,
            285,
            387,
            449,
            488,
            307,
            348,
            456,
            60,
            472,
            255,
            343,
            351,
            112,
            514,
            48,
            126,
            127,
            389,
            390,
            487,
            504,
            73,
            74,
            80,
            7,
            219,
            133,
            132,
            19,
            47,
            122,
            45,
            236,
            274,
            241,
            98,
            100,
            99,
            177,
            176,
            440,
            470,
            394,
            46,
            111,
            113,
            359,
            155,
            156,
            208,
            195,
            325,
            322,
            396,
            94,
            428,
            193,
            191,
            53,
            88,
            461,
            374,
            331,
            492,
            33,
            283,
            294,
            302,
            185,
            197,
            188,
            69,
            70,
            439,
            437,
            145,
            278,
            463,
            464,
            306,
            412,
            513,
            438,
            443,
            242,
            284,
            49,
            268,
            411,
            117,
            118,
            180,
            181,
            179,
            82,
            90,
            91,
            263,
            336,
            340,
            335,
            415,
            186,
            184,
            258,
            158,
            225,
            65,
            500,
            135,
            515,
            269,
            493,
            257,
            256,
            21,
            436,
            18,
            20,
            297,
            251,
            226,
            420,
            271,
            124,
            131,
            129,
            379,
            125,
            466,
            110,
            397,
            369,
            238,
            254,
            405,
            409,
            494,
            30,
            454,
            97,
            96,
            516,
            496,
            71,
            187,
            324,
            323,
            280,
            432,
            85,
            477,
            224,
            403,
            296,
            206,
            205,
            6,
            42,
            266,
            89,
            149,
            249,
            28,
            221,
            167,
            318,
            363,
            272,
            270,
            475,
            382,
            104,
            106,
            311,
            483,
            264,
            209,
            384,
            347,
            212,
            87,
            482,
            41,
            229,
            231,
            400,
            276,
            517,
            388,
            425,
            418,
            424,
            34,
            107,
            210,
            39,
            123,
            196,
            232,
            244,
            175,
            275,
            316,
            341,
            393,
            361,
            12,
            81,
            83,
            279,
            460,
            75,
            486,
            352,
            344,
            391,
            8,
            303,
            373,
            174,
            102,
            103,
            253,
            246,
            144,
            423,
            476,
            447,
            451,
            54,
            408,
            448,
            452,
            282,
            429,
            502,
            345,
            520,
            422,
            327,
            310,
            116,
            115,
            108,
            365,
            364,
            146,
            182,
            233,
            154,
            157,
            152,
            153,
            442,
            93,
            56,
            147,
            295,
            305,
            495,
            183,
            356,
            273,
            52,
            338,
            40,
            227,
            23,
            499,
            190,
            240,
            441,
            79,
            402,
            248,
            421,
            4,
            51,
            342,
            366,
            200,
            362,
            38,
            31,
            201,
            15,
            491,
            406,
            395,
            211,
            333,
            259,
            485,
            160,
            468,
            105,
            380,
            252,
            458,
            101,
            261,
            3,
            37,
            25,
            377,
            13,
            289,
            287,
            417,
            416,
            35,
            150,
            479,
            399,
            350,
            29,
            521,
            286,
            450,
            234,
            503,
            63,
            523,
            260,
            497,
            202,
            178,
            245,
            57,
            407,
            109,
            121,
            524,
            319,
            130,
            199,
            237,
            419,
            44,
            320,
            213,
            148,
            61,
            290,
            16,
            17,
            355,
            375,
            309,
            43,
            114,
            203,
            77,
            214,
            173,
            67,
            128,
            66,
            235,
            76,
            78,
            501,
            385,
            55,
            217,
            36,
            216,
            293,
            473,
            478,
            22,
            490,
            367,
            414,
            207,
            198,
            471,
            262,
            59,
            204,
            267,
            480,
            250,
            518,
            465,
            228,
            457,
            1,
            334,
            372,
            511,
            455,
            5,
            314,
            2,
            291,
            383,
            136,
            50,
        ]
        self.mdl_impl = mdl_impl
        self.mdl_impl.eval()

    def forward(self, x):
        # probs = self.mdl_impl.extract_features(
        # x, padding_mask=None)[0]
        # mask = torch.arange(x.shape[-1],device=x.device).expand(len(x), x.shape[-1])
        # # padding_mask = mask < padding.unsqueeze(1)
        # padding_mask = mask < padding.unsqueeze(1)
        padding_mask = torch.zeros_like(x).bool()
        probs = self.mdl_impl.extract_features(x, padding_mask=padding_mask)[0]
        tar_prob = torch.zeros_like(probs)
        tar_prob[..., self.mappings] = probs
        return tar_prob, None


def beats_iter2():
    return BeatsWrapper(Path(__file__).parent / "BEATs_iter2_finetuned_on_AS2M_cpt1.pt")


def beats_iter3plus():
    return BeatsWrapper(Path(__file__).parent / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")


def beats_iter3():
    return BeatsWrapper(Path(__file__).parent / "BEATs_iter3_finetuned_on_AS2M_cpt1.pt")
