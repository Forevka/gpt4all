import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import localdocs
import mysettings

MySettingsTab {
    title: qsTr("LocalDocs Plugin (BETA)")
    contentItem: ColumnLayout {
        id: root
        spacing: 10

        property alias collection: collection.text
        property alias folder_path: folderEdit.text

        FolderDialog {
            id: folderDialog
            title: "Please choose a directory"
            currentFolder: StandardPaths.writableLocation(StandardPaths.HomeLocation)
            onAccepted: {
                root.folder_path = selectedFolder
            }
        }

        RowLayout {
            Layout.fillWidth: true
            height: collection.height + 20
            spacing: 10
            MyTextField {
                id: collection
                width: 225
                horizontalAlignment: Text.AlignJustify
                color: theme.textColor
                placeholderText: qsTr("Collection name...")
                placeholderTextColor: theme.mutedTextColor
                ToolTip.text: qsTr("Name of the collection to add (Required)")
                ToolTip.visible: hovered
                Accessible.role: Accessible.EditableText
                Accessible.name: collection.text
                Accessible.description: ToolTip.text
                function showError() {
                    collection.placeholderTextColor = theme.textErrorColor
                }
                onTextChanged: {
                    collection.placeholderTextColor = theme.mutedTextColor
                }
            }

            MyDirectoryField {
                id: folderEdit
                Layout.fillWidth: true
                text: root.folder_path
                placeholderText: qsTr("Folder path...")
                placeholderTextColor: theme.mutedTextColor
                ToolTip.text: qsTr("Folder path to documents (Required)")
                ToolTip.visible: hovered
                function showError() {
                    folderEdit.placeholderTextColor = theme.textErrorColor
                }
                onTextChanged: {
                    folderEdit.placeholderTextColor = theme.mutedTextColor
                }
            }

            MyButton {
                id: browseButton
                text: qsTr("Browse")
                onClicked: {
                    folderDialog.open();
                }
            }

            MyButton {
                id: addButton
                text: qsTr("Add")
                Accessible.role: Accessible.Button
                Accessible.name: text
                Accessible.description: qsTr("Add button")
                onClicked: {
                    var isError = false;
                    if (root.collection === "") {
                        isError = true;
                        collection.showError();
                    }
                    if (root.folder_path === "" || !folderEdit.isValid) {
                        isError = true;
                        folderEdit.showError();
                    }
                    if (isError)
                        return;
                    LocalDocs.addFolder(root.collection, root.folder_path)
                    root.collection = ""
                    root.folder_path = ""
                    collection.clear()
                }
            }
        }

        ScrollView {
            id: scrollView
            Layout.fillWidth: true
            Layout.bottomMargin: 20
            clip: true
            contentHeight: 300
            ScrollBar.vertical.policy: ScrollBar.AlwaysOn

            background: Rectangle {
                color: theme.backgroundLighter
            }

            ListView {
                id: listView
                model: LocalDocs.localDocsModel
                boundsBehavior: Flickable.StopAtBounds
                delegate: Rectangle {
                    id: item
                    width: listView.width
                    height: buttons.height + 20
                    color: index % 2 === 0 ? theme.backgroundLight : theme.backgroundLighter
                    property bool removing: false

                    Text {
                        id: collectionId
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.margins: 20
                        text: collection
                        elide: Text.ElideRight
                        color: theme.textColor
                        width: 200
                    }

                    Text {
                        id: folderId
                        anchors.left: collectionId.right
                        anchors.margins: 20
                        anchors.verticalCenter: parent.verticalCenter
                        text: folder_path
                        elide: Text.ElideRight
                        color: theme.textColor
                    }

                    Item {
                        id: buttons
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 20
                        width: Math.max(removeButton.width, busyIndicator.width)
                        height: Math.max(removeButton.height, busyIndicator.height)
                        MyButton {
                            id: removeButton
                            anchors.centerIn: parent
                            text: qsTr("Remove")
                            visible: !item.removing && installed
                            onClicked: {
                                item.removing = true
                                LocalDocs.removeFolder(collection, folder_path)
                            }
                        }
                        MyBusyIndicator {
                            id: busyIndicator
                            anchors.centerIn: parent
                            visible: item.removing || !installed
                        }
                    }
                }
            }
        }

        GridLayout {
            id: gridLayout
            Layout.fillWidth: true
            columns: 3
            rowSpacing: 10
            columnSpacing: 10

            Rectangle {
                Layout.row: 0
                Layout.column: 0
                Layout.fillWidth: true
                Layout.columnSpan: 3
                height: 1
                color: theme.dialogBorder
            }

            Rectangle {
                Layout.row: 3
                Layout.column: 0
                Layout.fillWidth: true
                Layout.columnSpan: 3
                height: 1
                color: theme.dialogBorder
            }

            // This is here just to stretch out the third column
            Rectangle {
                Layout.row: 3
                Layout.column: 2
                Layout.fillWidth: true
                height: 1
                color: theme.dialogBorder
            }

            Label {
                id: chunkLabel
                Layout.row: 1
                Layout.column: 0
                color: theme.textColor
                text: qsTr("Document snippet size (characters):")
            }

            MyTextField {
                id: chunkSizeTextField
                Layout.row: 1
                Layout.column: 1
                ToolTip.text: qsTr("Number of characters per document snippet.\nNOTE: larger numbers increase likelihood of factual responses, but also result in slower generation.")
                ToolTip.visible: hovered
                text: MySettings.localDocsChunkSize
                validator: IntValidator {
                    bottom: 1
                }
                onEditingFinished: {
                    var val = parseInt(text)
                    if (!isNaN(val)) {
                        MySettings.localDocsChunkSize = val
                        focus = false
                    } else {
                        text = MySettings.localDocsChunkSize
                    }
                }
            }

            Label {
                id: contextItemsPerPrompt
                Layout.row: 2
                Layout.column: 0
                color: theme.textColor
                text: qsTr("Document snippets per prompt:")
            }

            MyTextField {
                Layout.row: 2
                Layout.column: 1
                ToolTip.text: qsTr("Best N matches of retrieved document snippets to add to the context for prompt.\nNOTE: larger numbers increase likelihood of factual responses, but also result in slower generation.")
                ToolTip.visible: hovered
                text: MySettings.localDocsRetrievalSize
                validator: IntValidator {
                    bottom: 1
                }
                onEditingFinished: {
                    var val = parseInt(text)
                    if (!isNaN(val)) {
                        MySettings.localDocsRetrievalSize = val
                        focus = false
                    } else {
                        text = MySettings.localDocsRetrievalSize
                    }
                }
            }

            Label {
                id: warningLabel
                Layout.row: 1
                Layout.column: 2
                Layout.rowSpan: 2
                Layout.maximumWidth: 520
                Layout.alignment: Qt.AlignTop
                color: theme.textErrorColor
                wrapMode: Text.WordWrap
                text: qsTr("Warning: Advanced usage only. Values too large may cause localdocs failure, extremely slow responses or failure to respond at all. Roughly speaking, the {N chars x N snippets} are added to the model's context window. More info <a href=\"https://docs.gpt4all.io/gpt4all_chat.html#localdocs-beta-plugin-chat-with-your-data\">here.</a>")
                onLinkActivated: function(link) { Qt.openUrlExternally(link) }
            }

            MyButton {
                id: restoreDefaultsButton
                Layout.row: 4
                Layout.column: 1
                Layout.columnSpan: 2
                Layout.fillWidth: true
                text: qsTr("Restore Defaults")
                Accessible.role: Accessible.Button
                Accessible.name: text
                Accessible.description: qsTr("Restores the settings dialog to a default state")
                onClicked: {
                    MySettings.restoreLocalDocsDefaults();
                }
            }
        }
    }
}
