"""
test_pii_detector.py — Unit tests for pii_detector.py

Strategy:
  - All tests work entirely with synthetic OCRWord lists.
  - No real images, no Tesseract binary needed.
  - Each test targets a specific entity type with positive and negative cases.
  - Tests verify: entity_type, bbox accuracy, context_snippet safety, confidence.

Run with:
    pytest backend/tests/unit/test_pii_detector.py -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import unittest

from app.models.findings import BoundingBox, EntityType, Finding, OCRWord
from app.services.pii_detector import detect_pii, _luhn_check, _overlaps_any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(text: str, x: int = 0, y: int = 0, conf: float = 0.90, page: int = 0) -> OCRWord:
    """Create a simple OCRWord with a small auto-sized bounding box."""
    return OCRWord(
        text=text,
        bbox=BoundingBox(x=x, y=y, width=len(text) * 8, height=20),
        confidence=conf,
        page=page,
    )


def _words_from_sentence(sentence: str, y: int = 10, conf: float = 0.90) -> list[OCRWord]:
    """Split a sentence into OCRWord objects with sequential x positions."""
    words = []
    x = 10
    for token in sentence.split():
        w = _word(token, x=x, y=y, conf=conf)
        words.append(w)
        x += len(token) * 8 + 10  # small gap between words
    return words


def _get_findings_of_type(findings: list[Finding], etype: EntityType) -> list[Finding]:
    return [f for f in findings if f.entity_type == etype]


# ===========================================================================
# Luhn algorithm tests
# ===========================================================================

class TestLuhnCheck(unittest.TestCase):

    def test_valid_visa(self):
        # Classic Visa test number
        self.assertTrue(_luhn_check("4111111111111111"))

    def test_valid_mastercard(self):
        self.assertTrue(_luhn_check("5500005555555559"))

    def test_invalid_number(self):
        self.assertFalse(_luhn_check("1234567890123456"))

    def test_short_number_invalid(self):
        self.assertFalse(_luhn_check("123"))

    def test_with_spaces(self):
        # Spaces are stripped internally
        self.assertTrue(_luhn_check("4111 1111 1111 1111"))


# ===========================================================================
# _overlaps_any helper
# ===========================================================================

class TestOverlapsAny(unittest.TestCase):

    def test_no_overlap(self):
        spans = [(0, 10), (20, 30)]
        self.assertFalse(_overlaps_any(11, 15, spans))

    def test_overlap_left(self):
        spans = [(10, 20)]
        self.assertTrue(_overlaps_any(5, 15, spans))

    def test_overlap_right(self):
        spans = [(10, 20)]
        self.assertTrue(_overlaps_any(15, 25, spans))

    def test_contained_within(self):
        spans = [(5, 25)]
        self.assertTrue(_overlaps_any(10, 20, spans))

    def test_adjacent_no_overlap(self):
        spans = [(0, 10)]
        self.assertFalse(_overlaps_any(10, 20, spans))


# ===========================================================================
# EMAIL detection
# ===========================================================================

class TestEmailDetection(unittest.TestCase):

    def test_detects_plain_email(self):
        words = _words_from_sentence("Email: user@example.com Thank you")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0].text, "user@example.com")

    def test_detects_gmail(self):
        words = _words_from_sentence("Contact: example@gmail.com for queries")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertEqual(len(emails), 1)
        self.assertIn("gmail.com", emails[0].text)

    def test_detects_email_without_context_keyword(self):
        # Email is unambiguous; no context keyword required
        words = _words_from_sentence("john.doe@hotel.in")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertEqual(len(emails), 1)

    def test_no_false_positive_on_plain_text(self):
        words = _words_from_sentence("The price is 1500 per night booking confirmed")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertEqual(len(emails), 0)

    def test_context_snippet_does_not_contain_email(self):
        words = _words_from_sentence("Contact: contact@stay.com for support")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertEqual(len(emails), 1)
        # The context snippet must not contain the actual email value
        self.assertNotIn("contact@stay.com", emails[0].context_snippet)
        self.assertIn("[REDACTED]", emails[0].context_snippet)

    def test_finding_confidence_in_range(self):
        words = _words_from_sentence("Email: admin@test.org")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertTrue(0.0 <= emails[0].confidence <= 1.0)


# ===========================================================================
# PHONE_NUMBER detection
# ===========================================================================

class TestPhoneDetection(unittest.TestCase):

    def test_detects_with_phone_keyword(self):
        words = _words_from_sentence("Phone: 9876543210 is the contact")
        findings = detect_pii(words)
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        self.assertEqual(len(phones), 1)
        self.assertIn("9876543210", phones[0].text)

    def test_detects_with_mob_keyword(self):
        words = _words_from_sentence("Mob: 9876543210")
        findings = detect_pii(words)
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        self.assertEqual(len(phones), 1)

    def test_detects_international_format_with_keyword(self):
        words = _words_from_sentence("Tel: +91-9876543210 please call")
        findings = detect_pii(words)
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        self.assertGreater(len(phones), 0)

    def test_no_detection_bare_number_no_context(self):
        # A plain 10-digit number with NO phone keyword should NOT be flagged
        words = _words_from_sentence("Invoice number 9876543210 total amount 5000")
        findings = detect_pii(words)
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        self.assertEqual(len(phones), 0)

    def test_no_detection_short_number(self):
        words = _words_from_sentence("Phone: 123 is not a valid number")
        findings = detect_pii(words)
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        self.assertEqual(len(phones), 0)


# ===========================================================================
# CARD_NUMBER detection
# ===========================================================================

class TestCardNumberDetection(unittest.TestCase):

    def test_detects_valid_visa_card(self):
        words = _words_from_sentence("Card: 4111111111111111 used for payment")
        findings = detect_pii(words)
        cards = _get_findings_of_type(findings, EntityType.CARD_NUMBER)
        self.assertEqual(len(cards), 1)

    def test_detects_spaced_card(self):
        # "4111 1111 1111 1111" — OCR may output as separate tokens
        parts = ["4111", "1111", "1111", "1111"]
        words = _words_from_sentence(" ".join(parts))
        findings = detect_pii(words)
        cards = _get_findings_of_type(findings, EntityType.CARD_NUMBER)
        self.assertEqual(len(cards), 1)

    def test_luhn_invalid_card_not_detected(self):
        # Invalid card — fails Luhn check
        words = _words_from_sentence("Card 1234567890123456 payment")
        findings = detect_pii(words)
        cards = _get_findings_of_type(findings, EntityType.CARD_NUMBER)
        self.assertEqual(len(cards), 0)

    def test_short_number_not_detected(self):
        words = _words_from_sentence("Order 12345 processed")
        findings = detect_pii(words)
        cards = _get_findings_of_type(findings, EntityType.CARD_NUMBER)
        self.assertEqual(len(cards), 0)


# ===========================================================================
# PASSPORT_NUMBER detection
# ===========================================================================

class TestPassportDetection(unittest.TestCase):

    def test_detects_with_keyword(self):
        words = _words_from_sentence("Passport No: AB1234567 issued")
        findings = detect_pii(words)
        passports = _get_findings_of_type(findings, EntityType.PASSPORT_NUMBER)
        self.assertEqual(len(passports), 1)
        self.assertIn("AB1234567", passports[0].text)

    def test_detects_single_letter_prefix(self):
        words = _words_from_sentence("Passport: Z9876543 valid till 2030")
        findings = detect_pii(words)
        passports = _get_findings_of_type(findings, EntityType.PASSPORT_NUMBER)
        self.assertEqual(len(passports), 1)

    def test_standalone_with_nearby_context(self):
        # Pattern without keyword label but "passport" appears in window
        words = _words_from_sentence("Please submit passport copy AB1234567 at reception")
        findings = detect_pii(words)
        passports = _get_findings_of_type(findings, EntityType.PASSPORT_NUMBER)
        self.assertGreater(len(passports), 0)

    def test_no_false_positive_random_alphanumeric(self):
        words = _words_from_sentence("Order ID QZ123 confirmed for your stay")
        findings = detect_pii(words)
        passports = _get_findings_of_type(findings, EntityType.PASSPORT_NUMBER)
        self.assertEqual(len(passports), 0)


# ===========================================================================
# BOOKING_REFERENCE detection
# ===========================================================================

class TestBookingReferenceDetection(unittest.TestCase):

    def test_detects_pnr_keyword(self):
        words = _words_from_sentence("PNR: AX72K9 flight booking confirmed")
        findings = detect_pii(words)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(bookings), 1)
        self.assertIn("AX72K9", bookings[0].text)

    def test_detects_booking_ref_keyword(self):
        words = _words_from_sentence("Booking Ref: HTL7892X thank you")
        findings = detect_pii(words)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(bookings), 1)

    def test_detects_confirmation_no(self):
        words = _words_from_sentence("Confirmation No: CONF123 valid")
        findings = detect_pii(words)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(bookings), 1)

    def test_no_detection_without_keyword(self):
        # A bare 6-char code without context should NOT fire
        words = _words_from_sentence("Transaction AX72K9 total 5000")
        findings = detect_pii(words)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(bookings), 0)

    def test_no_detection_short_code(self):
        words = _words_from_sentence("PNR: AB1 not valid")
        findings = detect_pii(words)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(bookings), 0)


# ===========================================================================
# ROOM_NUMBER detection
# ===========================================================================

class TestRoomNumberDetection(unittest.TestCase):

    def test_detects_room_no(self):
        words = _words_from_sentence("Room No: 1208 check in today")
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        self.assertEqual(len(rooms), 1)
        self.assertIn("1208", rooms[0].text)

    def test_detects_suite(self):
        words = _words_from_sentence("Suite: 305 executive floor")
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        self.assertEqual(len(rooms), 1)

    def test_detects_rm_abbreviation(self):
        words = _words_from_sentence("Rm No: 42 maintenance required")
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        self.assertEqual(len(rooms), 1)

    def test_no_detection_without_keyword(self):
        # A standalone number without room keyword should NOT fire
        words = _words_from_sentence("Floor 12 total occupancy 200")
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        self.assertEqual(len(rooms), 0)


# ===========================================================================
# ADDRESS detection
# ===========================================================================

class TestAddressDetection(unittest.TestCase):

    def test_detects_address_label(self):
        words = _words_from_sentence("Address: 42 MG Road Bengaluru Karnataka 560001")
        findings = detect_pii(words)
        addresses = _get_findings_of_type(findings, EntityType.ADDRESS)
        self.assertEqual(len(addresses), 1)

    def test_detects_flat_no_label(self):
        words = _words_from_sentence("Flat No: 204 Green Valley Apartments Delhi")
        findings = detect_pii(words)
        addresses = _get_findings_of_type(findings, EntityType.ADDRESS)
        self.assertEqual(len(addresses), 1)

    def test_no_detection_without_keyword(self):
        words = _words_from_sentence("Total amount 5000 due by checkout")
        findings = detect_pii(words)
        addresses = _get_findings_of_type(findings, EntityType.ADDRESS)
        self.assertEqual(len(addresses), 0)


# ===========================================================================
# Finding object integrity
# ===========================================================================

class TestFindingIntegrity(unittest.TestCase):

    def test_to_safe_dict_redacts_text(self):
        words = _words_from_sentence("Email: safe@test.com customer")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertTrue(len(emails) > 0)
        safe = emails[0].to_safe_dict()
        self.assertEqual(safe["text"], "[REDACTED]")
        self.assertNotIn("safe@test.com", str(safe))

    def test_to_dict_contains_text(self):
        words = _words_from_sentence("Email: raw@test.com here")
        findings = detect_pii(words)
        emails = _get_findings_of_type(findings, EntityType.EMAIL)
        self.assertTrue(len(emails) > 0)
        d = emails[0].to_dict()
        self.assertEqual(d["text"], "raw@test.com")

    def test_bbox_is_valid(self):
        words = _words_from_sentence("Room No: 808 check out")
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        self.assertTrue(len(rooms) > 0)
        bbox = rooms[0].bbox
        self.assertIsInstance(bbox, BoundingBox)
        self.assertGreaterEqual(bbox.x, 0)
        self.assertGreaterEqual(bbox.y, 0)
        self.assertGreater(bbox.width, 0)
        self.assertGreater(bbox.height, 0)

    def test_context_snippet_safe_to_log(self):
        """context_snippet must never contain the raw sensitive value."""
        sentences = [
            ("Phone: 9876543210 guest checkout", EntityType.PHONE_NUMBER),
            ("PNR: AX72K9 confirmed", EntityType.BOOKING_REFERENCE),
            ("Room No: 1208 allocated", EntityType.ROOM_NUMBER),
        ]
        for sentence, etype in sentences:
            words = _words_from_sentence(sentence)
            findings = detect_pii(words)
            typed = _get_findings_of_type(findings, etype)
            if typed:
                f = typed[0]
                self.assertNotIn(f.text, f.context_snippet,
                                 msg=f"Raw text found in context_snippet for {etype}")
                self.assertIn("[REDACTED]", f.context_snippet)

    def test_multiple_entities_in_one_image(self):
        """Multiple PII types in a single OCR result — all should be found."""
        tokens = (
            "Guest John Phone: 9876543210 Email: john@hotel.com "
            "Room No: 1208 Booking Ref: HTL7892X"
        )
        words = _words_from_sentence(tokens)
        findings = detect_pii(words)

        entity_types = {f.entity_type for f in findings}
        self.assertIn(EntityType.PHONE_NUMBER, entity_types)
        self.assertIn(EntityType.EMAIL, entity_types)
        self.assertIn(EntityType.ROOM_NUMBER, entity_types)
        self.assertIn(EntityType.BOOKING_REFERENCE, entity_types)

    def test_empty_words_returns_empty(self):
        findings = detect_pii([])
        self.assertEqual(findings, [])

    def test_no_pii_words_returns_empty(self):
        words = _words_from_sentence("Thank you for staying with us have a nice day")
        findings = detect_pii(words)
        self.assertEqual(findings, [])

    def test_findings_sorted_by_position(self):
        """Findings must be ordered top-to-bottom, left-to-right."""
        # Place phone at y=80 and email at y=10 — email should appear first
        phone_words = _words_from_sentence("Phone: 9876543210", y=80)
        email_words = _words_from_sentence("Email: z@example.com", y=10)
        words = email_words + phone_words
        findings = detect_pii(words)

        if len(findings) >= 2:
            # First finding must have smaller or equal y
            self.assertLessEqual(findings[0].bbox.y, findings[1].bbox.y)

    def test_source_page_propagated(self):
        words = [
            _word("Room", x=10, y=10, page=2),
            _word("No:", x=80, y=10, page=2),
            _word("507", x=130, y=10, page=2),
        ]
        findings = detect_pii(words)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        if rooms:
            self.assertEqual(rooms[0].source_page, 2)

    def test_no_number_only_false_positive(self):
        """Plain numbers without any keyword context must not be flagged."""
        words = _words_from_sentence(
            "Invoice 1001 Amount 5000 Items 3 Total 15000 Tax 2700"
        )
        findings = detect_pii(words)
        # None of these plain numbers should trigger PII
        phones = _get_findings_of_type(findings, EntityType.PHONE_NUMBER)
        rooms = _get_findings_of_type(findings, EntityType.ROOM_NUMBER)
        bookings = _get_findings_of_type(findings, EntityType.BOOKING_REFERENCE)
        self.assertEqual(len(phones), 0)
        self.assertEqual(len(rooms), 0)
        self.assertEqual(len(bookings), 0)


if __name__ == "__main__":
    unittest.main()
