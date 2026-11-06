from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_app.handbook import HandbookKnowledgeBase
from rag_app.api import app


@pytest.fixture
def handbook_text():
    return """
    University Student Handbook

    Page 1
    Attendance Policy: Students must attend at least 80% of all classes. In cases of illness, students must provide a medical certificate within three days.

    Page 2
    Library Services: Students may borrow up to five books for a maximum period of 21 days. Late returns incur a R15 fee per day.

    Page 3
    Fees and Payments: All tuition and fees are due before the first week of semester. Payment plans are available for approved applicants.

    Page 4
    Academic Integrity: Plagiarism is a serious offence and may lead to disciplinary action. Students must cite all sources properly.

    Page 5
    Examinations: Students must pass all major assessments to qualify for the final exam. Deferred exams require an approved application.

    Page 6
    Student Support: The counselling office provides academic and wellbeing support. Appointments can be booked online or in person.

    Page 7
    IT and Internet Access: The campus Wi-Fi is available for enrolled students. Students should use only academic resources and respect digital policies.

    Page 8
    Residence Rules: Residents must comply with quiet hours between 10pm and 6am and register guests before 8pm.

    Page 9
    Graduation Requirements: Students must complete all required modules and maintain a minimum average of 60 percent.

    Page 10
    Disciplinary Process: Faculty may issue warnings, academic probation, or suspension for repeated misconduct.
    """


@pytest.fixture
def index(tmp_path, handbook_text):
    handbook_path = tmp_path / "handbook.txt"
    handbook_path.write_text(handbook_text, encoding="utf-8")
    knowledge = HandbookKnowledgeBase(data_dir=tmp_path)
    knowledge.index_documents(handbook_path)
    return knowledge


def test_answer_question_returns_relevant_answer(index):
    answer, source = index.answer_question("What is the attendance requirement?")
    assert "80%" in answer
    assert source is not None


def test_answer_question_handles_missing_information(index):
    answer, source = index.answer_question("What is the hidden asteroid policy?")
    assert "could not find" in answer.lower()
    assert source in {"N/A", "Not found"}


@pytest.mark.parametrize(
    "question, expected_phrase",
    [
        ("How many books can a student borrow?", "five books"),
        ("When are fees due?", "first week of semester"),
        ("What happens if a student plagiarises?", "disciplinary action"),
        ("What is needed for a deferred exam?", "approved application"),
        ("How can students book counselling support?", "online or in person"),
        ("What is the campus Wi-Fi policy?", "enrolled students"),
        ("What are the quiet hours?", "10pm and 6am"),
        ("What average is required to graduate?", "60 percent"),
        ("What are the possible disciplinary actions?", "warning"),
        ("What can students do if they are ill?", "medical certificate"),
        ("What is the attendance policy?", "80%"),
        ("What is the late return fee for library books?", "15"),
    ],
)
def test_knowledge_base_covers_core_handbook_topics(index, question, expected_phrase):
    answer, _ = index.answer_question(question)
    assert expected_phrase.lower() in answer.lower()


def test_api_returns_json_response(index):
    app.state.index = index
    client = TestClient(app)
    response = client.post("/ask", json={"question": "What is the attendance requirement?"})
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "source" in payload
    assert "80%" in payload["answer"]


def test_api_rejects_invalid_json(index):
    app.state.index = index
    client = TestClient(app)
    response = client.post("/ask", data='not-json')
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]
