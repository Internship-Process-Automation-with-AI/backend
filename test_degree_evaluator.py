#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to debug degree evaluator matching issues.
"""

import os
import sys

# Add the current directory (src) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import after path setup
from llm.degree_evaluator import DegreeEvaluator  # noqa: E402


def test_degree_evaluator():
    """Test the degree evaluator with the actual job data."""

    evaluator = DegreeEvaluator()

    # Test with the actual job data from the certificate
    degree_program = "Insinööri (AMK), tieto- ja viestintätekniikka"

    # Get degree info
    degree_info = evaluator.get_degree_info(degree_program)
    print(f"Degree: {degree_info['name']}")
    print(f"Total possible roles: {len(degree_info['relevant_roles'])}")
    print(f"Total possible industries: {len(degree_info['relevant_industries'])}")

    # Test each position
    positions = [
        {
            "title": "Varaosavastaava",
            "responsibilities": "Varaosatietojen perustaminen, päivittäminen ja esittämistavan yhdenmukaistaminen Ylöjärven ja Tampereen tehtaiden järjestelmissä, liittyen hankitun ostojärjestelmän käytettävyyteen. Osallistuminen ostojärjestelmän asiakaskohtaisten raporttien määrittelyyn vastaamaan kunnossapidon tarpeita. Perehtyminen uuteen kalvonleikkuukoneeseen, sen varaosatietojen perustaminen, tekninen tuki kunnossapidossa ja tuotannon henkilöstölle. Varaosa-, kunnossapito- ja huoltosopimusasioissa yhdyshenkilönä kotimaisiin ja ulkomaisiin toimittajiin. Kehitystehtäviä, tavoitteena tuotteen laadunparannus ja edullisempi kunnossapito.",
        },
        {
            "title": "Tekninen Ostaja",
            "responsibilities": "Uusia tuotantolaitteita: tasolasin palastelulinja, esikäsittelylinja, hiontalinja ja karkaisulaitos. Varaosien hankintaa em. laitteisiin, osallistumista operoinnin ja huollon koulutuksiin ja määräaikaishuoltoihin. Osallistuminen kunnossapidon töihin ja perehdyttämiseen erityisosaamisalueillani. Yhteydet kotimaisiin ja ulkomaisiin toimittajiin kunnossapitöissä ja varaosa-asioissa. Suorat ostot osien valmistajilta ja osien valmistuttaminen parantavaan ja edullisempaan kunnossapitoon pyrkien.",
        },
        {
            "title": "Kunnossapidon Esimies",
            "responsibilities": "Kunnossapidon esimiestehtävät ja varaosahankinnat. Murskalinjan ja hionnan jäähdytysveden puhdistustekniikan hankinta uudelle esikäsittelylinjalle. Kaavionlevityslaitteen hankinta ja kaavionvalmistuksen tilojen suunnittelu. Karkaisulaitosten kunnossapito- ja varaosa-asiat.",
        },
        {
            "title": "Kehitysinsinööri",
            "responsibilities": "Esikäsittelykoneiden ja painokoneen teknisiä määrittelyjä sekä hankintasopimusten valmisteluja. Perehtyminen painovärin kuivaimiin (IR/UV). Tuotantolinjojen layout -suunnittelua ja rakennusteknisten lähtötietojen määrittelyä. Laitteiden asennusvalvontaa, esikäsittelykoneiden ja silkkipainokoneen hyväksyntätestien tekeminen, osallistuminen koulutuksiin ja kunnossapitotöihin perehdyttäen kunnossapidon ja tuotannon henkilöstölle osaamisalueillani. Esikäsittelylinjojen timanttityökalujen ja varaosien hankinta kotimaisilta ja ulkomaisilta toimittajilta.",
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING EACH POSITION")
    print("=" * 80)

    for i, position in enumerate(positions, 1):
        print(f"\n--- Position {i}: {position['title']} ---")

        # Test the relevance calculation
        relevance_level, multiplier = evaluator.calculate_relevance_score(
            degree_program, position["title"], position["responsibilities"], ""
        )

        print(f"Relevance Level: {relevance_level}")
        print(f"Multiplier: {multiplier}")

        # Let's manually check what keywords should match
        combined_text = f"{position['title']} {position['responsibilities']}".lower()

        print(f"\nCombined text: {combined_text[:100]}...")

        # Check for specific keywords that should match
        expected_matches = [
            "varaosahallinta",
            "ostojärjestelmät",
            "tekniset määrittelyt",
            "järjestelmäintegraatio",
            "laitteiden hallinta",
            "projektinhallinta",
            "kunnossapito",
            "kehitysinsinööri",
            "asennusvalvonta",
            "tietojärjestelmät",
        ]

        print("\nExpected keyword matches:")
        for keyword in expected_matches:
            if keyword in combined_text:
                print(f"  ✅ '{keyword}' - FOUND")
            else:
                print(f"  ❌ '{keyword}' - NOT FOUND")

        # Check for partial matches
        print("\nChecking for partial matches:")
        for role in degree_info["relevant_roles"]:
            role_lower = role.lower()
            if role_lower in combined_text:
                print(f"  ✅ '{role_lower}' - EXACT MATCH")
            elif any(word in combined_text for word in role_lower.split()):
                print(f"  🔶 '{role_lower}' - PARTIAL MATCH")


if __name__ == "__main__":
    test_degree_evaluator()
