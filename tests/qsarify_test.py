from qsarify import main


def test_qsarify(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello from qsarify!"
