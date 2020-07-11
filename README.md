# tensorflow-project-template

- GitHub では既存の Repository を Template として、新しい Repository を作成することができる
  - https://dev.classmethod.jp/articles/github-template-repository/

## Python のバージョン

- Mac OS python 3.7.4 で動作確認
- pyenv により、python のバージョンを変更する
  - https://qiita.com/Kohey222/items/19eb9b3cbcec9b176625

## 仮想環境の設定

- 仮想環境名

  - env-tensorflow-project-template-TODO-change-this-name
    - setup.sh

- venv: Python 仮想環境管理
  - https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e

```
$ python3 -m venv [newenvname]
$ source [newenvname]/bin/activate
$ ([newenvname])\$ deactivate
```

## Jupyter の設定

- Jupyter で複数カーネルを簡単に選択するための設定

  - https://qiita.com/tomochiii/items/8b937f15c79a0c3eae0e

```
# 利用可能なカーネルの表示
$ jupyter kernelspec list

# 仮想環境をアクティベート
$ source [newenvname]/bin/activate
# カーネルを追加
$ ipython kernel install --user --name=[newenvname] --display-name=[newenvname]
```

- カーネルを削除

```
$ jupyter kernelspec uninstall [newenvname]
```

- JupyterLab のターミナルフォントを変更する

  - https://thr3a.hatenablog.com/entry/20190916/1568618516

  1. 上のメニューバーより「Setting」→「Advanced Settings Editor」をクリック
  2. 左がデフォルト設定値一覧で、右にユーザーがオーバーライドしたい設定を JSON で記述する。
  3. User Preferences に以下を追記する。

```
{
    "fontFamily": "Monaco"
}
```

## graphviz のインストール

- tf.keras.utils.plot_model(model, show_shapes=True) で必要

```
brew install graphviz
```

## dataset

- TensorFlow で使えるデータセット機能が強かった話
  - https://qiita.com/Suguru_Toyohara/items/820b0dad955ecd91c7f3
