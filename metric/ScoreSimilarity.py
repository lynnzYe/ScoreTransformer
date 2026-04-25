from enum import IntEnum
from typing import List, Tuple

import music21
import numpy as np
from music21 import articulations, converter, expressions, stream
from numba import jit


class ScoreErrors(IntEnum):
    Clef = 0
    KeySignature = 1
    TimeSignature = 2
    NoteDeletion = 3
    NoteInsertion = 4
    NoteSpelling = 5
    NoteDuration = 6
    StemDirection = 7
    Beams = 8
    Tie = 9
    StaffAssignment = 10
    Voice = 11

    TrillTP = 12
    TrillFP = 13
    TrillTN = 14
    TrillFN = 15

    StaccatoTP = 16
    StaccatoFP = 17
    StaccatoTN = 18
    StaccatoFN = 19

    GraceTP = 20
    GraceFP = 21
    GraceTN = 22
    GraceFN = 23


def score_alignment(score_a, score_b):
    """Compare two musical scores.

    Parameters:

        aScore/bScore: music21.stream.Score objects

    Return value:

        (path, d):
            path is a list of tuples containing pairs of matching offsets
            d is the alignment matrix
    """

    def score_to_pitch_offset_list(s):
        """Convert a piano score into a list of tuples containing pitches

        Parameters
        ----------
            s: a music21.Stream containing two music21.stream.PartStaff

        Returns
        -------
            np.ndarray, shape=(N,) offsets
            np.ndarray, shape=(N, 128) a binary array to indicate the presence of a
                pitch onset at that location
        """
        my_dict = {}
        for n in s.flatten().notes:
            if n.isChord:
                raise NotImplementedError("Chords are not supported")
            if n.offset not in my_dict:
                my_dict[n.offset] = np.zeros(128, dtype=np.uint16)
            my_dict[n.offset][n.pitch.midi] = 1
        offsets = list(my_dict.keys())
        pitches = list(my_dict.values())
        return np.array(offsets), np.array(pitches)

    @jit(nopython=True, cache=True)
    def cost_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        d = np.zeros((s.shape[0] + 1, t.shape[0] + 1))
        d[1:, 0] = np.inf
        d[0, 1:] = np.inf
        for j in range(t.shape[0]):
            for i in range(s.shape[0]):
                d[i + 1, j + 1] = min(d[i, j + 1], d[i + 1, j], d[i, j]) + np.bitwise_xor(s[i], t[j]).sum()
        return d

    offsets_a, pitches_a = score_to_pitch_offset_list(score_a)
    offsets_b, pitches_b = score_to_pitch_offset_list(score_b)
    if not pitches_a.shape[0] or not pitches_b.shape[0]:
        return [(0, 0)]
    d = cost_matrix(pitches_a, pitches_b)

    # Uncomment for alignment path visual
    # d_img = d
    # d_img[np.isinf(d_img)] = d_img[np.logical_not(np.isinf(d_img))].max()
    # d_img = (d_img / d_img.max() * 255).astype(np.uint8)

    i, j = (d.shape[0] - 1, d.shape[1] - 1)
    path = []
    while i and j:
        # d_img[i - 1, j - 1] = 255
        path.insert(0, (offsets_a[i - 1], offsets_b[j - 1]))

        idx = np.argmin([d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]])
        if idx == 0:
            i = i - 1
        elif idx == 1:
            j = j - 1
        else:
            i, j = i - 1, j - 1
    return path


METER_MAP_PLUS = {  # Limit meter vocab for simplicity & tactus coherence
    2: {
        1: [1, 4],
        2: [4, 4],
        3: [3, 4],
        4: [4, 4],
    },
    4: {
        1: [1, 4],
        2: [4, 4],
        3: [3, 4],
        4: [4, 4],
        5: [5, 4],
        6: [6, 8],
        8: [4, 4],
    },
    8: {
        3: [3, 8],
        4: [4, 4],
        5: [5, 4],
        6: [6, 8],
        8: [4, 4],
        9: [9, 8],
        12: [12, 8],
    },
    16: {
        6: [6, 8],
        8: [4, 4],
        9: [3, 4],
        12: [12, 8],
        16: [4, 4],
        24: [12, 8],
    }
}


def score_similarity(
        estScore: stream.Score | str,
        gtScore: stream.Score | str,
        partMapping={0: "right", 1: "left"},
        full=False,
) -> dict[str, float | int | None]:
    """Compare two musical scores.

    Parameters
    ----------
    estScore: music21.stream.Score
        Predicted piano score or path to its location.
    gtScore: music21.stream.Score
        Ground truth piano score or path to its location.
    partMapping: dict[int, str]
        A dictionary mapping part indices to hands. The keys are the part indices
        (0-based) and the values are either "right" or "left".
    full: bool
        Whether to run a full comparison including all attributes or not.
        Will run significantly faster if not full.
    Returns
    -------
    Dict[str, float | int | None]:
        A dictionary containing the number of errors for each type of error.
    """
    if isinstance(estScore, str):
        estScore = converter.parse(estScore, forceSource=True)
    estScore = estScore.expandRepeats().stripTies(preserveVoices=False)
    if isinstance(gtScore, str):
        gtScore = converter.parse(gtScore, forceSource=True)
    gtScore = gtScore.expandRepeats().stripTies(preserveVoices=False)
    assert isinstance(estScore, stream.Stream)
    assert isinstance(gtScore, stream.Stream)

    def score_to_list(aScore):
        """Convert a piano score into a list of tuples

        Parameters
        ----------
        aScore: a music21.Stream containing two music21.stream.PartStaff

        Returns
        -------
        List[Tuple[float, int, music21 object]]
            offset is a real number indicating the offset of an object in music21 terms
            staff is an integer indicating the staff (0 = top, 1 = bottom)
            object is a music21 object
        """

        # Classes to consider
        CLASSES = (music21.bar.Barline, music21.note.Note, music21.chord.Chord)

        def stream_to_list(s, hand) -> List[Tuple[float, Tuple[int, music21.base.Music21Object]]]:
            elements_with_offsets = {}
            for el in s.recurse():
                if isinstance(el, CLASSES):
                    offset = el.getOffsetInHierarchy(s)
                    elements_with_offsets.setdefault(offset, []).append((hand, el))
            return sorted(elements_with_offsets.items())

        def flatten_stream(s: music21.stream.Stream):
            new_stream = music21.stream.Stream()
            for el in s.flatten():
                if isinstance(el, (music21.note.Note, music21.chord.Chord)):
                    new_stream.insert(el)
            for el in s.recurse():
                if isinstance(el, (music21.stream.Measure, music21.bar.Barline)):
                    new_stream.insert(el.getOffsetInHierarchy(s), music21.bar.Barline())
            return new_stream

        # We only report staff/hand scores if we are sure which part corresponds to
        # which part
        validParts = True
        parts = aScore.getElementsByClass(
            [music21.stream.PartStaff, music21.stream.Part]
        )
        topStaffList = []
        bottomStaffList = []

        for i, part in enumerate(parts):
            if i not in partMapping:
                # All notes for both scores will be merged into the top staff later
                topStaffList += stream_to_list(flatten_stream(part), 0)
                validParts = False
            elif partMapping[i] == "right":
                topStaffList += stream_to_list(flatten_stream(part), 0)
            else:
                bottomStaffList += stream_to_list(flatten_stream(part), 1)

        elTimes = []
        elList = []
        tIterator = iter(topStaffList)
        bIterator = iter(bottomStaffList)
        tEl = next(tIterator, None)
        bEl = next(bIterator, None)

        while tEl or bEl:
            if not tEl:
                elTimes.append(bEl[0])
                elList.append(bEl[1])
                bEl = next(bIterator, None)
            elif not bEl:
                elTimes.append(tEl[0])
                elList.append(tEl[1])
                tEl = next(tIterator, None)
            else:
                if tEl[0] < bEl[0]:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1])
                    tEl = next(tIterator, None)
                elif tEl[0] > bEl[0]:
                    elTimes.append(bEl[0])
                    elList.append(bEl[1])
                    bEl = next(bIterator, None)
                else:
                    elTimes.append(tEl[0])
                    elList.append(tEl[1] + bEl[1])
                    tEl = next(tIterator, None)
                    bEl = next(bIterator, None)
        if elTimes and elList:
            elTimes, elList = zip(*sorted(zip(elTimes, elList), key=lambda x: x[0]))

        return np.array(elTimes), elList, validParts

    def count_objects(a_set):
        """Count objects in a set

        Parameters
        ----------
        a_set: list of tuples (staff, object)
            staff is an integer indicating the staff (1 = top, 2 = bottom)
            object is a music21 object

        Returns
        -------
            int: the numbers of objects in the set
        """

        count = 0
        for obj in a_set:
            if isinstance(obj[1], music21.note.Note):
                count += 1
            elif isinstance(obj[1], music21.chord.Chord):
                count += len(obj[1].pitches)
        return count

    def compare_sets(a_set, b_set):
        """Compare two sets of concurrent musical objects.

        Parameters
        ----------
        a_set/b_set: list of tuples (staff, object)
            staff is an integer indicating the staff (1 = top, 2 = bottom)
            object is a music21 object

        Returns
        -------
            np.ndarray: An array with the differences between the two sets
                (see definition of errors below)
        """

        def pitchComp(pA, pB):
            """Compare two pitches
            In contrast to music21.pitch.Pitch.__eq__, this function allows
            no accidental to match a natural accidental.
            """
            if (pA.octave == pB.octave) and (pA.step == pB.step) and (pA.microtone == pB.microtone):
                if pA.accidental == pB.accidental:
                    return True
                if pA.accidental is None and getattr(pB.accidental, 'name', None) == 'natural':
                    return True
                if getattr(pA.accidental, 'name', None) == 'natural' and pB.accidental is None:
                    return True
            return False

        def findEnharmonicEquivalent(note, a_set):
            """Find the first enharmonic equivalent in a set

            Parameters
            ----------
            note: a music21.note.Note object
            a_set: list of tuples (staff, object)
                staff is an integer indicating the staff (0 = top, 1 = bottom)
                object is a music21 object

            Returns
            -------
                int: index of the first enharmonic equivalent of note in a_set -1 otherwise
            """
            for i, obj in enumerate(a_set):
                if isinstance(obj[1], music21.note.Note) and obj[1].pitch.ps == note.pitch.ps:
                    return i
            return -1

        def compare_obj(aObj, bObj):
            """
            Compare two music21 objects and return True if they are similar, False otherwise.

            Parameters
            ----------
                aObj: The first music21 object to compare.
                bObj: The second music21 object to compare.

            Returns
            -------
                bool: True if the two objects are similar, False otherwise.
            """
            if aObj == bObj:
                return True
            if isinstance(aObj, music21.key.Key) or isinstance(aObj, music21.key.KeySignature):
                return aObj.sharps == bObj.sharps
            if type(aObj) != type(bObj):
                return False
            if isinstance(aObj, music21.note.Note):
                if (
                        pitchComp(aObj.pitch, bObj.pitch)
                        and abs(aObj.duration.quarterLength - bObj.duration.quarterLength)
                        < 1e-3
                        and aObj.stemDirection == bObj.stemDirection
                ):
                    return True
            if isinstance(aObj, music21.bar.Barline):
                return True
            if isinstance(aObj, music21.stream.Measure):
                return True
            if isinstance(aObj, music21.clef.Clef):
                return True
            if isinstance(aObj, music21.meter.TimeSignature):
                if (
                        aObj.numerator == bObj.numerator
                        and aObj.beatCount == bObj.beatCount
                ):
                    return True
            return False

        def get_beams(note_obj: music21.note.Note):
            return '_'.join(['-'.join([b.type, b.direction]) if b.direction else b.type for b in note_obj.beams])

        def get_grace(note_obj: music21.note.Note):
            return note_obj.duration.isGrace

        def get_staccato(note_obj: music21.note.Note):
            if not hasattr(note_obj, 'articulations'):
                return False
            staccatos = (articulations.Staccatissimo, articulations.Staccato)
            return any(isinstance(e, staccatos) for e in note_obj.articulations)

        def get_clef(note_obj: music21.note.Note):
            context = note_obj.getContextByClass('Clef')
            return context.name if context is not None else ''

        def get_key_sig(note_obj: music21.note.Note):
            keyObj = (note_obj.getContextByClass('Key') or note_obj.getContextByClass('KeySignature'))
            return keyObj.sharps if keyObj else 0

        def get_tie(note_obj: music21.note.Note):
            return note_obj.tie.type if note_obj.tie is not None else ''

        def get_time_sig(note_obj: music21.note.Note):
            context = note_obj.getContextByClass(music21.meter.TimeSignature)
            # return context.numerator / context.denominator if context else ""

            # Take a relaxed time signature evaluation instead:
            # e.g. 2/4 == 4/4, 6/4 = 6/8
            if not context: return ''
            if context.denominator not in METER_MAP_PLUS or context.numerator not in METER_MAP_PLUS[
                context.denominator]:
                return context.numerator / context.denominator
            relaxed_meter = METER_MAP_PLUS[context.denominator][context.numerator]
            return relaxed_meter[0] / relaxed_meter[1]

        def get_trill(note_obj: music21.note.Note):
            if not hasattr(note_obj, 'expressions'):
                return False
            trills = (
                expressions.Trill,
                expressions.InvertedMordent,
                expressions.Mordent,
                expressions.Turn,
            )
            return any(isinstance(e, trills) for e in note_obj.expressions)

        def get_voice(note_obj: music21.note.Note):
            context = note_obj.getContextByClass('Voice')
            return context.id if context is not None else '1'

        errors = np.zeros((len(ScoreErrors.__members__)), int)

        a = a_set.copy()
        b = b_set.copy()

        def track_note_errors(a, b):
            if not isinstance(a, music21.note.Note):
                return
            if not isinstance(b, music21.note.Note):
                return
            if abs(a.duration.quarterLength - b.duration.quarterLength) > 1e-3:
                errors[ScoreErrors.NoteDuration] += 1

            if a.stemDirection != b.stemDirection:
                errors[ScoreErrors.StemDirection] += 1

            if get_beams(a) != get_beams(b):
                errors[ScoreErrors.Beams] += 1

            if get_grace(a):
                if get_grace(b):
                    errors[ScoreErrors.GraceTP] += 1
                else:
                    errors[ScoreErrors.GraceFN] += 1
            else:
                if get_grace(b):
                    errors[ScoreErrors.GraceFP] += 1
                else:
                    errors[ScoreErrors.GraceTN] += 1

            if get_staccato(a):
                if get_staccato(b):
                    errors[ScoreErrors.StaccatoTP] += 1
                else:
                    errors[ScoreErrors.StaccatoFN] += 1
            else:
                if get_staccato(b):
                    errors[ScoreErrors.StaccatoFP] += 1
                else:
                    errors[ScoreErrors.StaccatoTN] += 1

            if get_tie(a) != get_tie(b):
                errors[ScoreErrors.Tie] += 1

            if get_trill(a):
                if get_trill(b):
                    errors[ScoreErrors.TrillTP] += 1
                else:
                    errors[ScoreErrors.TrillFN] += 1
            else:
                if get_trill(b):
                    errors[ScoreErrors.TrillFP] += 1
                else:
                    errors[ScoreErrors.TrillTN] += 1

            # Only run slow checks if full is set to True
            if full:
                # if get_clef(a) != get_clef(b): # ignore clef 
                #     errors[ScoreErrors.Clef] += 1
                if get_key_sig(a) != get_key_sig(b):
                    errors[ScoreErrors.KeySignature] += 1
                if get_time_sig(a) != get_time_sig(b):
                    errors[ScoreErrors.TimeSignature] += 1
                # if get_voice(a) != get_voice(b): # and voice
                #     errors[ScoreErrors.Voice] += 1

        # Step 2: Find exact matches
        # We filter a and b by removing matches from b,
        # and building a new 'a' that does not include the matches
        a_temp = []
        for pair in a:
            for bPair in b:
                if pair[0] == bPair[0] and compare_obj(pair[1], bPair[1]):
                    track_note_errors(bPair[1], pair[1])
                    b.remove(bPair)
                    break
            else:
                a_temp.append(pair)
        a = a_temp

        # Step 3
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Note):
                for bObj in b:
                    if (
                            isinstance(bObj[1], music21.note.Note)
                            and pitchComp(bObj[1].pitch, obj[1].pitch)
                    ):
                        if bObj[0] != obj[0]:
                            errors[ScoreErrors.StaffAssignment] += 1
                        else:  # TODO: probably remove the else, only left here for backwards compatibility
                            track_note_errors(bObj[1], obj[1])
                        b.remove(bObj)
                        break
                else:
                    a_temp.append(obj)
            else:
                a_temp.append(obj)
        a = a_temp
        # Find enharmonic equivalents and report spelling mistakes and duration mistakes
        a_temp = []
        for obj in a:
            if isinstance(obj[1], music21.note.Note):
                idx = findEnharmonicEquivalent(obj[1], b)
                if idx != -1:
                    if b[idx][0] != obj[0]:
                        errors[ScoreErrors.StaffAssignment] += 1
                    track_note_errors(b[idx][1], obj[1])
                    del b[idx]
                    errors[ScoreErrors.NoteSpelling] += 1
                    continue
            a_temp.append(obj)
        a = a_temp
        errors[ScoreErrors.NoteInsertion] = count_objects(a)
        errors[ScoreErrors.NoteDeletion] += count_objects(b)
        return errors

    def get_set(elements, times, start, end):
        endIdx = np.searchsorted(times, end, side='left')
        set = []
        for el in elements[start:endIdx]:
            set += el
        return set, endIdx

    path = score_alignment(estScore, gtScore)

    aTimes, aElements, est_parts_valid = score_to_list(estScore)
    bTimes, bElements, gt_parts_valid = score_to_list(gtScore)
    if not (est_parts_valid and gt_parts_valid):
        for i, aElement in enumerate(aElements):
            for j, el in enumerate(aElement):
                aElements[i][j] = (0, el[1])
        for i, bElement in enumerate(bElements):
            for j, el in enumerate(bElement):
                bElements[i][j] = (0, el[1])
    errors = np.zeros((len(ScoreErrors.__members__)), float)
    aIdx, aEnd = 0, 0.0
    bIdx, bEnd = 0, 0.0
    for pair in path:
        if pair[0] != aEnd and pair[1] != bEnd:
            aEnd = pair[0]
            bEnd = pair[1]
            aSet, aIdx = get_set(aElements, aTimes, aIdx, aEnd)
            bSet, bIdx = get_set(bElements, bTimes, bIdx, bEnd)
            errors += compare_sets(aSet, bSet)
        elif pair[0] == aEnd:
            bEnd = pair[1]
        else:
            aEnd = pair[0]

    aSet, _ = get_set(aElements, aTimes, aIdx, float('inf'))
    bSet, _ = get_set(bElements, bTimes, bIdx, float('inf'))
    errors += compare_sets(aSet, bSet)

    results = {k: int(v) for k, v in zip(ScoreErrors.__members__.keys(), errors)}

    def update_f1(d, key):
        prec = d[key + "TP"] / (d[key + "TP"] + d[key + "FP"]) if d[key + "TP"] + d[key + "FP"] else None
        rec = d[key + "TP"] / (d[key + "TP"] + d[key + "FN"]) if d[key + "TP"] + d[key + "FN"] else None
        if prec is None or rec is None:
            f1 = None
        else:
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else None
        d.update({key + "Prec": prec, key + "Rec": rec, key + "F1": f1})
        return prec, rec, f1

    update_f1(results, "Trill")
    update_f1(results, "Staccato")
    update_f1(results, "Grace")

    results.update({"n_Note": len([n for n in gtScore.flatten().notes if not n.isRest])})
    if not (est_parts_valid and gt_parts_valid):
        results['StaffAssignment'] = None

    results_relative = {}
    n_notes = results["n_Note"]
    n_matched = results["n_Note"] - results["NoteDeletion"]
    for k, v in results.items():
        if v is None:
            results_relative[k] = None
        elif k in ("n_Note",
                   "TrillTP", "TrillFP", "TrillTN", "TrillFN", "TrillF1",
                   "StaccatoTP", "StaccatoFP", "StaccatoTN", "StaccatoFN", "StaccatoF1",
                   "GraceTP", "GraceFP", "GraceTN", "GraceFN", "GraceF1"):
            results_relative[k] = v
        elif k in ["NoteInsertion", "NoteDeletion"]:
            results_relative[k] = v / (n_notes or 1e-9)
        else:
            results_relative[k] = v / (n_matched or 1e-9)
    return results


def test_score_similarity_same():
    score_path = (
        "score_transformer/tokenization_tools/tokenizer/sample/input_score.musicxml"
    )
    expected = {
        "Clef": 0,
        "KeySignature": 0,
        "TimeSignature": 0,
        "NoteDeletion": 0,
        "NoteInsertion": 0,
        "NoteSpelling": 0,
        "NoteDuration": 0,
        "StemDirection": 0,
        "Beams": 0,
        "Tie": 0,
        "StaffAssignment": 0,
        "Voice": 0,
        "TrillTP": 0,
        "TrillFP": 0,
        "TrillTN": 39,
        "TrillFN": 0,
        "StaccatoTP": 0,
        "StaccatoFP": 0,
        "StaccatoTN": 39,
        "StaccatoFN": 0,
        "GraceTP": 0,
        "GraceFP": 0,
        "GraceTN": 39,
        "GraceFN": 0,
        "TrillPrec": None,
        "TrillRec": None,
        "TrillF1": None,
        "StaccatoPrec": None,
        "StaccatoRec": None,
        "StaccatoF1": None,
        "GracePrec": None,
        "GraceRec": None,
        "GraceF1": None,
        "n_Note": 39,
    }
    res = score_similarity(score_path, score_path, full=True)
    assert res == expected


def test_score_similarity_same_2():
    from score_transformer import score_to_tokens
    from score_transformer import tokens_to_score

    score_path = (
        "score_transformer/tokenization_tools/tokenizer/sample/input_score.musicxml"
    )
    # gen_path = "score_transformer/tokenization_tools/detokenizer/sample/generated_score.musicxml"
    expected = {
        "Clef": 0,
        "KeySignature": 0,
        "TimeSignature": 0,
        "NoteDeletion": 0,
        "NoteInsertion": 0,
        "NoteSpelling": 0,
        "NoteDuration": 0,
        "StemDirection": 0,
        "Beams": 0,
        "Tie": 0,
        "StaffAssignment": 0,
        "Voice": 0,
        "TrillTP": 0,
        "TrillFP": 0,
        "TrillTN": 39,
        "TrillFN": 0,
        "StaccatoTP": 0,
        "StaccatoFP": 0,
        "StaccatoTN": 39,
        "StaccatoFN": 0,
        "GraceTP": 0,
        "GraceFP": 0,
        "GraceTN": 39,
        "GraceFN": 0,
        "TrillPrec": None,
        "TrillRec": None,
        "TrillF1": None,
        "StaccatoPrec": None,
        "StaccatoRec": None,
        "StaccatoF1": None,
        "GracePrec": None,
        "GraceRec": None,
        "GraceF1": None,
        "n_Note": 39,
    }
    res = score_similarity(
        tokens_to_score(" ".join(score_to_tokens(score_path))), score_path
    )
    # assert res == expected


def test_score_similarity_different():
    res = score_similarity("/rhome/beyer/thesis/data/asap-dataset-master/Bach/Fugue/bwv_848/xml_score.musicxml",
                           "/rhome/beyer/thesis/data/asap-dataset-master/Bach/Fugue/bwv_860/xml_score.musicxml")
    print(res)


if __name__ == "__main__":
    test_score_similarity_same()
    test_score_similarity_same_2()
    test_score_similarity_different()
